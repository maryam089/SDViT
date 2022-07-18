
    for command in delete_incomplete launch
    do
    python -m domainbed.scripts.sweep ${command} --data_dir=/home/computervision1/DG_new_idea/domainbed/data/ \
    --output_dir=./domainbed/PACS_Output/ERM_ViT/backbone_DeitSmall/  --command_launcher multi_gpu --algorithms ERM_ViT  \
    --single_test_envs  --datasets PACS  --n_hparams 1 --n_trials 3  \
    --hparams """{\"backbone\":\"DeitSmall\",\"batch_size\":32,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0}"""
    done
