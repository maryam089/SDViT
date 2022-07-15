# ERM-SDViT
# Self-Distilled Vision Transformer for Domain Generalization

The code is build on the top of DomainBed: a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434).


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

## Train your own models:
- Step 1: Download the pretrained models on Imagenet, DeiT, CVT-21, T2T-ViT-14 (TBA)
- Step 2: Place the models in the path ./domainbed/pretrained_models/Model_name/
- Step 3: Run the followng commands:  

Training a single model with an indiviual target domain (test_env) id 0:

```sh
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/PACS/\
       --algorithm ERM\
       --dataset PACS\
       --test_env 0
```

Launching a sweep: Training model with all target domains

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/Sweep Output/path\
       --command_launcher multi_gpu
```
Launching a sweep on ViT Baselines:

```sh
./Baseline_sweep.sh
```
Launching a sweep on Our Proposed Model:

```sh
./Grid_Search_sweep.sh
```
Note: For above all commands change --dataset PACS for training on other datasets such as OfficeHome, VLCS, TerraIncognita and DomainNet 

## Model selection criteria
We computed results on the following model selection
* `IIDAccuracySelectionMethod`: A random subset from the input data of the training source domains.
## Results Using Pre-trained Models
To view the results using our pre-trained models:
- Step 1: Download the pretrained models uisng this link (TBA)
- Step 2: Run the following command to get outputs
````sh
python -m domainbed.scripts.collect_results\
       --input_dir=/Results/Dataset/Model/ --get_recursively True
````
Note: Replace the text with dataset and model names (e.g: Results/PACS/Model/ and so on....) to view results on various models.
## Test-Time Classifier Adjuster (T3A)
T3A is exploited in our proposed method as a complimentary approach, for details please refer to following instructions:
[T3A](https://github.com/matsuolab/T3A)

## Acknowledgment
Code is based on [T2T](https://github.com/yitu-opensource/T2T-ViT), [CVT](https://github.com/microsoft/CvT), [DeiT](https://github.com/facebookresearch/deit) repository and [TIMM](https://github.com/rwightman/pytorch-image-models) library. We thank the authors for releasing their codes

## License

This source code is released under the MIT license, included [here](LICENSE).
