# Self-Distilled Vision Transformer for Domain Generalization
[Maryam Sultana](https://scholar.google.com/citations?user=dKsfEyIAAAAJ&hl=en), [Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en), [M.Haris Khan](https://scholar.google.com/citations?user=ZgERfFwAAAAJ&hl=en), [Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en), and [Fahad Shahbaz Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao)

[Paper]() ([arXiv]()) (TBA)
> **Abstract:** *In recent past, several domain generalization (DG) methods have been proposed, showing encouraging performance, however, almost all of them build on convolutional neural networks (CNNs). There is little to no progress on studying the DG performance of vision transformers (ViTs), which are challenging the supremacy of CNNs on standard benchmarks, often built on i.i.d assumption. This renders the real-world deployment of ViTs doubtful. In this paper, we attempt to explore ViTs towards addressing the DG problem. Similar to CNNs, ViTs also struggle in out-of-distribution scenarios and the main culprit is overfitting to source domains. Inspired by the modular architecture of ViTs, we propose a simple DG approach for ViTs, coined as self-distillation for ViTs. It reduces the overfitting to source domains by easing the learning of input-output mapping problem through curating non-zero entropy supervisory signals for intermediate transformer blocks. Further, it does not introduce any new parameters and can be seamlessly plugged into the modular composition of different ViTs. We empirically demonstrate notable performance gains with different DG baselines and various ViT backbones in five challenging datasets. Moreover, we report favorable performance against recent state-of-the-art DG methods. Our code along with pre-trained models are made available publicly.*

<p align="center">
     <img src="https://github.com/maryam089/SDViT/blob/main/Figures/proposed_method_git.png" > 
</p>
Proposed self-distillation in ViTs for domain generalization (ERM-SDViT). ViTs build upon a modular and a hierarchical architecture, where a model is comprised of $n$ intermediate blocks/layers f_{i} and a final classifier h. The 'Selector' chooses a random block from the range of intermediate blocks and makes a prediction after passing its classification token through the final classifier. This way the dark knowledge, as non-zero entropy signals, is distilled from the final classification token to the intermediate class tokens during training.


## News Updates
- SDViT pre-trained models will be available after (25/07/2022).


## Citation
If you find our work useful. Please consider giving a star :star: and cite our work. (TBA)
```bibtex
@InProceedings{
}
```

### Contents  
1) [Highlights](#Highlights) 
2) [Installation](#Installation)
3) [Datasets](#Datasets)
4) [Training Vision Transformer with Self-Distillation](#Training-Vision-Transformer-with-Self-Distillation)
5) [Pretrained Models](#Pretrained-Models)
5) [Evaluating for Domain Generalization](#Evaluating-for-Domain-Generalization)
6) [Visual Examples](#Visual-Examples)


### Highlights

<p align="center">
     <img src="https://github.com/maryam089/SDViT/blob/main/Figures/blockwise_accuracy_git.png" > 
</p>
In the Fig. above, we plot the block-wise accuracy of baseline (ERM-ViT) and our method (ERM-SDViT). Random sub-model distillation improves the accuracy of all blocks, in particular, the improvement is more pronounced for the earlier blocks. Besides later blocks, it also encourages earlier blocks to bank on transferable representations, yet discriminative representations. Since these earlier blocks manifest multiple discriminative feature pathways, we believe that they better facilitate the overall model towards capturing the semantics of the object class.


### Installation
To install conda env with conda, run the following command in your terminal:
```sh
conda env create -n ViT_DGbed --file ViT_DGbed.yml
```
Activate the conda environment:
```sh
conda activate ViT_DGbed
```
### Datasets:

```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data --dataset pacs
```
Note: for downloading other datasets change --dataset pacs with other datasets (e.g., vlcs, office_home, terra_incognita, domainnet).


### Model selection criteria
We computed results on the following model selection
* `IIDAccuracySelectionMethod`: A random subset from the input data of the training source domains.

### Training Vision Transformer with Self-Distillation

- Step 1: Download the pretrained models on Imagenet, such as [CVT-21](https://onedrive.live.com/?authkey=%21AMXesxbtKwsdryE&cid=56B9F9C97F261712&id=56B9F9C97F261712%2115008&parId=56B9F9C97F261712%2115004&o=OneUp), [T2T-ViT-14](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.5_T2T_ViT_14.pth.tar)
- Step 2: Place the models in the path ./domainbed/pretrained_models/Model_name/
- Step 3: Run the followng commands:  

Launching a sweep on ViT Baselines:

```sh
./Baseline_sweep.sh
```
Launching a sweep on SDViT Model:

```sh
./Grid_Search_sweep.sh
```
Note: For above all commands change --dataset PACS for training on other datasets such as OfficeHome, VLCS, TerraIncognita and DomainNet and backbone to CVTSmall or T2T14.

### Pretrained Models
Pretrained weights for ViT models trained (TBA)

| Dataset   |                                               Baseline (ERM-ViT)                                              |                                                                                             Ours (ERM-SDViT) | 
|:----------|:----------------------------------------------------------------------------------------------------:|-------------------------------------------------------------------------------------------------:|
| PACS      |   [Link]()    |   [Link]() | 
| VLCS | [Link]() | [Link]() | 
| OfficeHome    |  [Link]()   |  [Link]() | 
| TerraIncognita    |  [Link]()   |  [Link]() | 
| DomainNet    |  [Link]()   |   [Link]() | 


<hr />
<hr />

### Evaluating for Domain Generalization
#Results Using Pre-trained Models
To view the results using our pre-trained models:
- Step 1: Download the pretrained models uisng the links in above Table
- Step 2: Run the following command to get outputs
````sh
python -m domainbed.scripts.collect_results\
       --input_dir=/Results/Dataset/Model/Backbone/ --get_recursively True
````
Note: Replace the text with dataset and model names (e.g: Results/PACS/ERM-ViT/DeiT-Small/ and so on....) to view results on various models. Test-Time Classifier Adjuster (T3A) is exploited in our proposed method as a complimentary approach, for details please refer to following instructions:
[T3A](https://github.com/matsuolab/T3A)

**Ablation Results:**

1. Accuracy on three Backbone Networks using PACS dataset.
![results](Figures/Table_2.png)
**SOTA Results:**
2. Accuracy on three Backbone Networks using five benchmark datasets in comparison with DG SOTA.
![results](Figures/Table_2.png)

### Visual Examples

<p align="center">
     <img src="https://github.com/maryam089/SDViT/blob/main/Figures/PACS_git.png" > 
</p>
Comparison of attention maps between the baseline ERM-ViT and our proposed ERM-SDViT (backbone: DeiT-Small) on four target domains of PACS dataset. 
<p align="center">
     <img src="https://github.com/maryam089/SDViT/blob/main/Figures/Attentions_VLCS_OH.png" > 
</p>
Comparison of attention maps between the baseline ERM-ViT and our proposed ERM-SDViT (backbone: DeiT-Small) on four target domains of VLCS and OfficeHome dataset.
<p align="center">
     <img src="https://github.com/maryam089/SDViT/blob/main/Figures/Attentions_DomainNet.png" > 
</p>
Comparison of attention maps between the baseline ERM-ViT and our proposed ERM-SDViT (backbone: DeiT-Small) on six target domains of DomainNet dataset.


### Acknowledgment
The code is build on the top of DomainBed: a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434). ViT Code is based on [T2T](https://github.com/yitu-opensource/T2T-ViT), [CVT](https://github.com/microsoft/CvT), [DeiT](https://github.com/facebookresearch/deit) repository and [TIMM](https://github.com/rwightman/pytorch-image-models) library. We thank the authors for releasing their codes.

### License

This source code is released under the MIT license, included [here](LICENSE).
