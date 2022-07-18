# Self-Distilled Vision Transformer for Domain Generalization
[Maryam Sultana](https://scholar.google.com/citations?user=dKsfEyIAAAAJ&hl=en), [Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en), [M.Haris Khan](https://scholar.google.com/citations?user=ZgERfFwAAAAJ&hl=en), [Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en), and [Fahad Shahbaz Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao)
[Paper]() ([arXiv]()) (TBA)
> **Abstract:** *In recent past, several domain generalization (DG) methods have been proposed, showing encouraging performance, however, almost all of them build on convolutional neural networks (CNNs). There is little to no progress on studying the DG performance of vision transformers (ViTs), which are challenging the supremacy of CNNs on standard benchmarks, often built on i.i.d assumption. This renders the real-world deployment of ViTs doubtful. In this paper, we attempt to explore ViTs towards addressing the DG problem. Similar to CNNs, ViTs also struggle in out-of-distribution scenarios and the main culprit is overfitting to source domains. Inspired by the modular architecture of ViTs, we propose a simple DG approach for ViTs, coined as self-distillation for ViTs. It reduces the overfitting to source domains by easing the learning of input-output mapping problem through curating non-zero entropy supervisory signals for intermediate transformer blocks. Further, it does not introduce any new parameters and can be seamlessly plugged into the modular composition of different ViTs. We empirically demonstrate notable performance gains with different DG baselines and various ViT backbones in five challenging datasets. Moreover, we report favorable performance against recent state-of-the-art DG methods. Our code along with pre-trained models are made available publicly.*
> 
## News Updates
- SDViT pre-trained models will be available after (25/07/2022).


## Citation
If you find our work useful. Please consider giving a star :star: and cite our work. (TBA)
```bibtex
@InProceedings{
}
```

### Contents  
1) [Contributions](#Contributions) 
2) [Target Transferability Vs Model Disparity](#Target-Transferability-Vs-Model-Disparity)
3) [Pretrained Targeted Generator](#Pretrained-Targeted-Generator) 
4) [Training](#Training)
5) [Evaluation](#Evaluation)
7) [Why Augmentations boost Transferability?](#Why-Augmentations-boost-Transferability)
8) [Why Ensemble of weak Models maximizes Transferability?](#Why-Ensemble-of-weak-Models-maximizes-Transferability)
9) [Generative Vs Iterative Attacks](#Generative-Vs-Iterative-Attacks)
     * [Key Developments made by Iterative Attacks](#Key-Developments-made-by-Iterative-Attacks)
     * [Key Developments made by Generative Attacks](#Key-Developments-made-by-Generative-Attacks)
10) [Tracking SOTA Targeted Transferability](#Tracking-SOTA-Targeted-Transferability) 
11) [What Can You Do?](#What-Can-You-Do)  
12) [Visual Examples](#Visual-Examples)


## Quick start
To install conda env with conda, run the following command in your terminal:
```sh
conda env create -n ViT_DGbed --file ViT_DGbed.yml
```
Activate the conda environment:
```sh
conda activate ViT_DGbed
```
## Download the datasets:

```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data --dataset pacs
```
Note: for downloading other datasets change --dataset pacs with other datasets (e.g., vlcs, office_home, terra_incognita, domainnet).


## Model selection criteria
We computed results on the following model selection
* `IIDAccuracySelectionMethod`: A random subset from the input data of the training source domains.
## Train SDViT models:
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

## Evaluation:
#Results Using Pre-trained Models
To view the results using our pre-trained models:
- Step 1: Download the pretrained models uisng this link (TBA)
- Step 2: Run the following command to get outputs
````sh
python -m domainbed.scripts.collect_results\
       --input_dir=/Results/Dataset/Model/Backbone/ --get_recursively True
````
Note: Replace the text with dataset and model names (e.g: Results/PACS/ERM-ViT/DeiT-Small/ and so on....) to view results on various models. Test-Time Classifier Adjuster (T3A)
T3A is exploited in our proposed method as a complimentary approach, for details please refer to following instructions:
[T3A](https://github.com/matsuolab/T3A)

## Acknowledgment
The code is build on the top of DomainBed: a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434). ViT Code is based on [T2T](https://github.com/yitu-opensource/T2T-ViT), [CVT](https://github.com/microsoft/CvT), [DeiT](https://github.com/facebookresearch/deit) repository and [TIMM](https://github.com/rwightman/pytorch-image-models) library. We thank the authors for releasing their codes.

## License

This source code is released under the MIT license, included [here](LICENSE).
