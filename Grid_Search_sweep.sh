
for lambda1 in 0.5 0.2 0.1
do
    for lambda2 in 3.0 5.0
    do
        for command in delete_incomplete launch
		    do
        python -m domainbed.scripts.sweep ${command} --data_dir=./domainbed/data/ \
        --output_dir=./domainbed/PACS_Output/ERM_SDViT_DeiT/sweep_RB_loss_${lambda1}_KL_Div_Temperature_${lambda2} --command_launcher multi_gpu --algorithms ERM_SDViT_DeiT  \
        --single_test_envs  --datasets PACS  --n_hparams 1 --n_trials 3  \
        --hparams """{\"batch_size\":32,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"RB_loss_weight\":$lambda1,\"KL_Div_Temperature\":$lambda2}"""
        done
    done
done
