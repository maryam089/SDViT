
for lambda1 in 0.5 0.2 0.1
do
    for lambda2 in 3.0 5.0
    do
        for command in delete_incomplete launch
		do
        python -m domainbed.scripts.sweep ${command} --data_dir=/home/computervision1/DG_new_idea/domainbed/data/ \
        --output_dir=./domainbed/PACS_Output/ERM_SDViT/backbone_DeitSmall/sweep_RB_loss_${lambda1}_KL_Div_Temperature_${lambda2} --command_launcher multi_gpu --algorithms ERM_SDViT  \
        --single_test_envs  --datasets PACS  --n_hparams 1 --n_trials 3  \
        --hparams """{\"backbone\":\"DeitSmall\",\"batch_size\":32,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"RB_loss_weight\":$lambda1,\"KL_Div_Temperature\":$lambda2}"""
        done
    done
done
