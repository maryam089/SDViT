
    for command in delete_incomplete launch
        do
    python -m domainbed.scripts.sweep ${command} --data_dir=./domainbed/data/ \
    --output_dir=./domainbed/PACS_Output/ERM_ViT_DeiT/  --command_launcher multi_gpu --algorithms ERM_ViT_DeiT  \
    --single_test_envs  --datasets PACS  --n_hparams 1 --n_trials 3  \
    --hparams """{\"batch_size\":32,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0}"""
     done
