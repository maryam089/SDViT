# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile
from collections import OrderedDict, defaultdict
from numbers import Number
import torchvision
import operator
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as pl
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import torch
import tqdm
from collections import Counter
# from domainbed import algorithms
# from domainbed.visiontransformer import VisionTransformer
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()


class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def print_separator():
    print("=" * 80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def accuracy(network, loader, weights, device,noise_sd=0.5,addnoise=False):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if(addnoise):
                x=x + torch.randn_like(x, device='cuda') * noise_sd
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            # print(p.shape)
            if len(p.shape)==1:
                p = p.reshape(1,-1)
            if p.size(1) == 1:

                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def two_model_analysis(network,network_comp, loader, weights, device,noise_sd=0.5,addnoise=False,env_name="env0"):
    correct = 0
    total = 0
    weights_offset = 0
    pred_cls_all=[]
    pred_comp_cls_all=[]
    y_all=[]
    all_x=[]
    network.eval()
    network_comp.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if(addnoise):
                x=x + torch.randn_like(x, device='cuda') * noise_sd
            p = network.predict(x)
            p_comp=network_comp.predict(x)
            pred_cls=torch.argmax(p, dim=1)
            pred_comp_cls=torch.argmax(p_comp, dim=1)
            pred_cls_all.append(pred_cls)
            pred_comp_cls_all.append(pred_comp_cls)
            y_all.append(y)
            all_x.append(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            # print(p.shape)
            if len(p.shape)==1:
                p = p.reshape(1,-1)
            if p.size(1) == 1:

                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
    print("predicted_cls")
    pred_cls_all=torch.cat(pred_cls_all,dim=0).cpu().numpy()
    pred_comp_cls_all=torch.cat(pred_comp_cls_all,dim=0).cpu().numpy()
    y_all=torch.cat(y_all,dim=0).cpu().numpy()
    all_x=torch.cat(all_x,dim=0)
    correct_classes=[]
    selected_images=[]
    for i in range(pred_cls_all.size):
        if(pred_cls_all[i]!=y_all[i] and pred_comp_cls_all[i]==y_all[i]):
            correct_classes.append(pred_comp_cls_all[i])
            torchvision.utils.save_image(all_x[i],'two_model_analysis_neg/test_env:'+env_name+'_cls:'+str(pred_comp_cls_all[i])+"_num_"+str(i)+'.png')
        else:
            correct_classes.append(-1)
    pred_corr=np.array([pred_cls_all==y_all]).astype('int')
    pred_comp_corr=np.array([pred_comp_cls_all==y_all]).astype('int')
    
    pred_only_correct=pred_corr-pred_comp_corr
    pred_only_correct[pred_only_correct<0]=0
    only_correct_in_algo=np.count_nonzero(pred_only_correct)
    print(np.count_nonzero(pred_only_correct))
    print(total)
    print(Counter(correct_classes))
    print("classes:",Counter(y_all))
    
    return only_correct_in_algo / total


def loss_ret(network, loader, weights, device,noise_sd=0.5,addnoise=False):
    correct = 0
    total = 0
    weights_offset = 0
    total_loss=0
    network.eval()
    counter=0
    with torch.no_grad():
        for x, y in loader:
            counter+=1
            x = x.to(device)
            y = y.to(device)
            # p,output_rb = network.network(x,flatness=True)
            if(addnoise):
                x=x + torch.randn_like(x, device='cuda') * noise_sd
            p = network.predict(x)
            
            loss=F.cross_entropy(p,y)
            # rb_loss = F.kl_div(
            #     F.log_softmax(output_rb / 6.0, dim=1),
            #     ## RB output cls token, original network output cls token
            #     F.log_softmax(p / 6.0, dim=1),
            #     reduction='sum',
            #     log_target=True
            # ) * (6.0 * 6.0) / output_rb.numel()

            total_loss+=loss
            # +0.5*rb_loss
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            # print(p.shape)
            if len(p.shape)==1:
                p = p.reshape(1,-1)
            if p.size(1) == 1:

                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return total_loss/(counter*1.0),correct / total

def confusionMatrix(network, loader, weights, device, output_dir, env_name, algo_name,args,algorithm_class,dataset,hparams):
    trials=3
    
    
    if algo_name is None:
        algo_name = type(network).__name__
    conf_mat_all=[]
    
    for i in range(trials):
        pretrained_path=args.pretrained
        pretrained_path=pretrained_path[:-14]+str(i)+pretrained_path[-13:]
        network = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams,pretrained_path) #args.pretrained
        network.to(device)
        correct = 0
        total = 0
        weights_offset = 0
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                p = network.predict(x)
                pred = p.argmax(1)
                y_true = y_true + y.to("cpu").numpy().tolist()
                y_pred = y_pred + pred.to("cpu").numpy().tolist()
                # print(y_true)
                # print("hashf")
                # print(y_pred)
                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset: weights_offset + len(x)]
                    weights_offset += len(x)
                batch_weights = batch_weights.to(device)
                if p.size(1) == 1:
                    # if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    # print('p hai ye', p.size(1))
                    correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total += batch_weights.sum().item()
        
        conf_mat = confusion_matrix(y_true, y_pred)
        # print(confusion_matrix(y_true, y_pred))
        conf_mat_all.append(conf_mat)
        print(conf_mat, 'cf_matrix')
    conf_mat=(conf_mat_all[0]+conf_mat_all[1]+conf_mat_all[2])/(trials*1.0)
    conf_mat=conf_mat.astype('int')
    print(conf_mat, 'cf_matrix_average')
    conf_mat=conf_mat/np.sum(conf_mat,axis=1,keepdims=True) #percentage calculator

    sn.set(font_scale=20)  # for label size
    plt.figure(figsize=(90, 90))
    # sn.heatmap(conf_mat, cbar=False,square=True, annot=True,annot_kws={"size": 90},fmt='d',xticklabels=['DG','EP','GF','GT','HR','HS','PR'],yticklabels=['DG','EP','GF','GT','HR','HS','PR'])  # font size
    ax=sn.heatmap(conf_mat, cmap="Blues", cbar=True,linewidths=4, square=True, annot=True,fmt='.1%',annot_kws={"size": 155},xticklabels=['0','1','2','3','4','5','6'],yticklabels=['0','1','2','3','4','5','6'])  # font size
    # ax=sn.heatmap(conf_mat, cbar=True, cmap="Blues",annot=True,fmt='.1%',annot_kws={"size": 90},linewidths=4, square = True, xticklabels=['0','1','2','3','4','5','6'],yticklabels=['0','1','2','3','4','5','6'])  # font size
    # ax=sn.heatmap(conf_mat, cbar=True, cmap="Blues",annot=True,fmt='.1%',annot_kws={"size": 90},linewidths=4, square = True, xticklabels=['0','1','2','3','4','5','6'],yticklabels=['0','1','2','3','4','5','6'])  # font size
    plt.yticks(rotation=0)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.axhline(y=0, color='k',linewidth=10)
    ax.axhline(y=conf_mat.shape[1], color='k',linewidth=10)

    ax.axvline(x=0, color='k',linewidth=10)
    ax.axvline(x=conf_mat.shape[1], color='k',linewidth=10)
    # plt.show()
    plt.savefig('Confusion_matrices/'+algo_name+env_name+'.png',bbox_inches='tight')


    

    return correct / total

# cmap='summer'


def TsneFeatures(network, loader, weights, device, output_dir, env_name, algo_name):
 

    correct = 0
    total = 0
    weights_offset = 0
    network.eval()
    Features=[[] for _ in range(12)]
    labels=[]
    if algo_name is None:
        algo_name = type(network).__name__
    try:
        Transnetwork = network.network
    except:
        Transnetwork = network.network_original
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            p,F = Transnetwork(x,return_feat=True)

            for i in range(len(F)):

                Features[i].append(F[i])
            labels.append(y)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
    labels=torch.cat(labels).cpu().detach().numpy()
    Features_all=[[] for _ in range(12)]
    for i in range(len(Features)):

        Features_all[i]=torch.cat(Features[i],dim=0).cpu().detach().numpy()
    # print(labels)
    print(labels.shape)
    
    name_conv=env_name

    # print(y)
    # print(len(y))
    # print(len(Features))
    # print(Features[0].shape)
    return Features_all,labels

def plot_block_accuracy2(network, loader, weights, device, output_dir, env_name, algo_name):
    # print(network)

    if algo_name is None:
        algo_name = type(network).__name__
    try:
        network = network.network
    except:
        network = network.network_original
    correct = [0] * len(network.blocks)
    total = [0] * len(network.blocks)
    weights_offset = [0] * len(network.blocks)

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p1 = network.acc_for_blocks(x)
            for count, p in enumerate(p1):
                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset[count]: weights_offset[count] + len(x)]
                    weights_offset[count] += len(x)
                batch_weights = batch_weights.to(device)
                # print(p.size, 'p size')
                # if p.size(1) == 1:
                if p.size(1) == 1:
                    correct[count] += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    # print('p hai ye', p.size(1))
                    correct[count] += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total[count] += batch_weights.sum().item()

    res = [i / j for i, j in zip(correct, total)]
    print(algo_name, ":", env_name, ":blockwise accuracies:", res)
    plt.plot(res)
    plt.title(algo_name)
    plt.xlabel('Block#')
    plt.ylabel('Acc')
    plt.ylim(0.0,1.0)
    plt.savefig(output_dir + "/" + algo_name + "_" + env_name + "_" + 'acc.png')
    return res


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
