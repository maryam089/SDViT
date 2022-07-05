# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#ResNet-18 True , data aug: True, normailization on
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import seaborn as sn
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
import copy



import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

def load_model(fname):
    dump = torch.load(fname)
    algorithm_class = algorithms.get_algorithm_class(dump["args"]["algorithm"])
    algorithm = algorithm_class(
        dump["model_input_shape"],
        dump["model_num_classes"],
        dump["model_num_domains"],
        dump["model_hparams"])
    algorithm.load_state_dict(dump["model_dict"])
    return algorithm

def visualizeEd(features: torch.Tensor, labels: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """
    # source_feature = source_feature.numpy()
    # target_feature = target_feature.numpy()

    # features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    # print(np.array(features).shape)
    # print(len(labels))
    labels=np.array(labels)
    features=np.array(features)
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    print("done")
    # domain labels, 1 represents source while 0 represents target
    
    # labels = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # tsne_data=np.vstack((X_tsne.T,labels)).T
    # tsne_df=pd.DataFrame(data=tsne_data,columns=("Dim_1","Dim_2","label"))
    # tsne_df.drop
    labelscls=labels%10
    palette = sn.color_palette("bright",7)
    sn.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=labelscls, legend='full', palette=palette)
    # sn.FacetGrid(tsne_df,hue="label",height=10).map(plt.scatter,'Dim_1','Dim_2')
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("TsneFIG2/clswise"+filename)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # tsne_data=np.vstack((X_tsne.T,labels)).T
    # tsne_df=pd.DataFrame(data=tsne_data,columns=("Dim_1","Dim_2","label"))
    # tsne_df.drop
    labelsd=labels//10
    labelsdom=labels*0
    labelsdom[labelsd==int(args.test_envs[0])]=1

    palette = sn.color_palette("bright", 2)
    sn.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=labelsdom, legend='full', palette=palette)
    # sn.FacetGrid(tsne_df,hue="label",height=10).map(plt.scatter,'Dim_1','Dim_2')
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("TsneFIG2/domainwise"+filename)



################################ Code required for RCERM ################################ 
from domainbed import queue_var
################################ Code required for RCERM ################################ 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="/home/computervision1/DG_new_idea/domainbed/data")
    parser.add_argument('--dataset', type=str, default="OfficeHome")
    parser.add_argument('--algorithm', type=str, default="Testing")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="test_env0_tr2")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--pretrained', type=str, default="/train_output/model.pkl")
    parser.add_argument('--pretrained_comp',type=str,default=None)
    parser.add_argument('--algo_name', type=str, default=None)
    parser.add_argument('--confusion_matrix', type=bool, default=False)
    parser.add_argument('--features', type=bool, default=False)
    parser.add_argument('--get_loss', type=bool, default=False)
    parser.add_argument('--noise_check', type=bool, default=False)
    parser.add_argument('--two_model_analy', type=bool, default=False)
    args = parser.parse_args()
 

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    # print("Environment:")
    # print("\tPython: {}".format(sys.version.split(" ")[0]))
    # print("\tPyTorch: {}".format(torch.__version__))
    # print("\tTorchvision: {}".format(torchvision.__version__))
    # print("\tCUDA: {}".format(torch.version.cuda))
    # print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    # print("\tNumPy: {}".format(np.__version__))
    # print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # print('HParams:')
    # for k, v in sorted(hparams.items()):
    #     print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # print('device:', device)
        
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    ### DEBUGGING    
#     print(dataset)
        
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset): #env is a domain
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    if args.algorithm=="Testing":
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams,args.pretrained)
    else:
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    if(args.two_model_analy):
        algorithm_comp=load_model(args.pretrained_comp)
        algorithm_comp.to(device)


    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    
        
    last_results_keys = None
    # for step in range(start_step, n_steps):
    step_start_time = time.time()
    minibatches_device = [(x.to(device), y.to(device))
        for x,y in next(train_minibatches_iterator)]
    if args.task == "domain_adaptation":
        uda_device = [x.to(device)
            for x,_ in next(uda_minibatches_iterator)]
    else:
        uda_device = None
    # step_vals = algorithm.update(minibatches_device, uda_device)
    checkpoint_vals['step_time'].append(time.time() - step_start_time)

    # for key, val in step_vals.items():
    #     checkpoint_vals[key].append(val)

    # if (step % checkpoint_freq == 0) or (step == n_steps - 1):
    results = {
        # 'step': step,
        # 'epoch': step / steps_per_epoch,
    }

    for key, val in checkpoint_vals.items():
        results[key] = np.mean(val)

    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    algo_name=args.algo_name
    name_conv=algo_name+str(args.test_envs)
    if(args.features):
        if(os.path.exists("tsne/TSVS/meta_"+name_conv+".tsv")):
            os.remove("tsne/TSVS/meta_"+name_conv+".tsv")
    Features_all=[[] for _ in range(12)]
    labels_all=[]
    for name, loader, weights in evals:
        env_name=name[:4]
        if(args.two_model_analy and int(name[3]) in args.test_envs and  "in" in name):
            # Comparing two models where one is correctly preicting and other is not
            corr_tot=misc.two_model_analysis(algorithm,algorithm_comp, loader, weights, device,env_name=env_name)

        elif(args.features):
            # Tsne
            if(int(name[3]) not in args.test_envs and  "in" in name):
                continue
            if(int(name[3]) in args.test_envs and  "out" in name):
                continue
            print(name)
            Features,labels=misc.TsneFeatures(algorithm, loader, weights, device,args.output_dir,env_name,algo_name)
            for blk in range(len(Features)):
                # Features_all[blk]+=Features[blk]

                # if(blk==11):
                #     print(Features[blk].shape)
                #     visualize(Features[blk][:1000], Features[blk][1000:],"check.jpg", source_color='r', target_color='b')
                #     print("Tsne done.....................................")
                with open("tsne/TSVS/records_"+name_conv+"_blk_"+str(blk)+".tsv", "a") as record_file:
                    for i in range(len(labels)):
                        Features_all[blk].append(Features[blk][i])
                        for j in range(len(Features[blk][i])):
                            
                            record_file.write(str(Features[blk][i][j]))
                            record_file.write("\t")
                        record_file.write("\n")
            # if (env_name[-1] in args.test_envs):
            with open("tsne/TSVS/meta_"+name_conv+".tsv", "a") as record_file:
                for i in range(len(labels)):
                    labels_all.append(int(str(env_name[-1])+str(labels[i])))

                    record_file.write(str(labels[i]))
                    record_file.write("\n")
        elif(args.get_loss  and  "in" in name):
            # Computing Flatness (comment gaussian noise with std for random normal scaling)
            loss_degr=[]
            x=list(np.arange(0.0,0.055,0.005))
            loss,acc=misc.loss_ret(algorithm, loader, weights, device)
            loss_degr.append(loss.item())
            accuracies=[]
            accuracies.append(acc)
            for rad in x:
                if rad==0:
                    continue
                total_loss=0
                tot_accuracy=0
                for j in range(10):

                    algo_cpy=copy.deepcopy(algorithm)
                    net=algo_cpy.network
                    Ws=copy.deepcopy(net.state_dict())
                    num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                    # print(num_trainable_params)
                    direction_vector = torch.randn(num_trainable_params)
                    unit_direction_vector = direction_vector / torch.norm(direction_vector)
                    unit_direction_vector*=rad

                    unit_direction_vector=torch.normal(0.0, float(rad), size=(sum(p.numel() for p in net.parameters() if p.requires_grad),))  #gaussian noise with std
                    i=0
                    for k,w in Ws.items():

                        w=w.to("cuda")
                        change=unit_direction_vector[i:i+w.numel()].reshape(w.shape).to("cuda")
                        w+=change
                        i=i+w.numel()
                    # print(Ws['head.weight'])
                    net.load_state_dict(Ws)
                    
                    loss_ch,acc=misc.loss_ret(algo_cpy, loader, weights, device)
                    loss_diff=loss_ch-loss
                    total_loss+=loss_ch
                    tot_accuracy+=acc
                total_loss/=10.0
                tot_accuracy/=10.0
                print(rad)
                print(total_loss)
                print(tot_accuracy)
                loss_degr.append(total_loss.item())
                accuracies.append(tot_accuracy)
                
            # plt.plot(x,loss_degr,linewidth=1.5,marker='x')
            # plt.xlabel('Gamma')
            # plt.ylabel('Flatness')
            # xticks = [10,20,30,40,50,60]
            # ticklabels = ['10','20','30','40','50','60']
            # xticks = [10,20]
            # ticklabels = ['10','20']
            # plt.xticks(xticks, ticklabels)
            # plt.savefig( 'Flatness/'+algo_name+"test_env"+str(args.test_envs)+'.png')
            print(algo_name,"_test_env_",str(args.test_envs))
            print(loss_degr)
            with open("flatness2.txt", "a") as record_file:    
                record_file.write("\n")   
                record_file.write("#"+algo_name+"test_env"+str(args.test_envs)+"train_env"+str(int(name[3])))
                record_file.write("\t")
                record_file.write("\n")
                record_file.write("l=")
                record_file.write(str(loss_degr))
                record_file.write("\n")
                record_file.write("ac=")
                record_file.write(str(accuracies))
                record_file.write("\n")


        elif(args.noise_check and  "in" in name):
            #Add noise to image
            loss_degr=[0]
            x=list(np.arange(0.0,4.0,0.4))
            losses=[]
            accuracies=[]
            for sigma in x:
                loss,acc=misc.loss_ret(algorithm, loader, weights, device,addnoise=True,noise_sd=sigma)
                losses.append(loss.item())
                accuracies.append(acc)
 
            print("#Losses",algo_name,"_test_env_",str(args.test_envs))
            print("l=",losses)
            with open("Loss_all_blks.txt", "a") as record_file:       
                record_file.write("#Losses, "+algo_name+"test_env"+str(args.test_envs))
                record_file.write("\t")
                record_file.write("\n")
                record_file.write("l=")
                record_file.write(str(losses))
                record_file.write("\n")
                record_file.write(str(x))
                record_file.write("\n")


            print("#Accuracies",algo_name,"_test_env_",str(args.test_envs))
            print("l=",accuracies)
            with open("Accuracy_all_blks.txt", "a") as record_file:       
                record_file.write("#Accuracies, "+algo_name+"test_env"+str(args.test_envs))
                record_file.write("\t")
                record_file.write("\n")
                record_file.write("l=")
                record_file.write(str(accuracies))
                record_file.write("\n")
                record_file.write(str(x))
                record_file.write("\n")

        elif (int(name[3]) in args.test_envs and  "in" in name):
            print("name",name)
            
            if(args.confusion_matrix):
                conf=misc.confusionMatrix(algorithm, loader, weights, device,args.output_dir,env_name,algo_name,args,algorithm_class,dataset,hparams)

               
            else:
                #plot blockwise accuracies for transformer
                block_acc=misc.plot_block_accuracy2(algorithm, loader, weights, device,args.output_dir,env_name,algo_name)
            


    if(args.features):
        #tsne 
        for i in range(12):

            visualizeEd(Features_all[i], labels_all,name_conv+str(i)+"blk.jpg", source_color='r', target_color='b')
            print("Tsne done.....................................")




