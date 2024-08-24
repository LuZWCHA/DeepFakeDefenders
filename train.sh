# python train.py --amp -e Resnet_101_NPR_deepfake_c2 --checkpoint /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/work_dir/classify/model_epoch_00000.pth --task_name Resnet_101_NPR_deepfake_c2 --model_type resnet101

# python train.py --amp -e Resnet_50_deepfake_c2 --task_name Resnet_50_deepfake_c2 --device "cuda:1" --model_type resnet50_ori
# python train.py --amp -e Resnet_50_NPR_ORI_deepfake_c2 --task_name Resnet_50_NPR_ORI_deepfake_c2 --device "cuda:0" --model_type resnet50_ori_npr --num_worker 4


# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /mnt/data/phase1/valset_label.txt \
#     -val_img /mnt/data/phase1/valset \
#     --amp -e EfficientNet_B2_NPR_GRAD_deepfake_c2 --task_name EfficientNet_B2_NPR_GRAD_deepfake_c2 --device "cuda:0" --model_type efficientnet_b2_npr_grad --num_worker 8 --batch_size 128

# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /mnt/data/phase1/valset_label.txt \
#     -val_img /mnt/data/phase1/valset \
#     --amp -e EfficientNet_B0_deepfake_c2 --task_name EfficientNet_B0_deepfake_c2 --device "cuda:0" --model_type efficientnet_b0 --num_worker 8 --batch_size 128

# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/phase1/testset_face_label.txt \
#     -val_img /nasdata2/private/lzhao/workspace/kaggle/DeepfakeDatasets/dataset \
#     --amp -e DINOV2_B14_deepfake_c2_nf --task_name DINOV2_B14_deepfake_c2_nf --device "cuda:0" --model_type dinov2_vitb14_nf --num_worker 16 --batch_size 32 \
#     --lr 1e-4

# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /mnt/data/phase1/valset_label.txt \
#     -val_img /mnt/data/phase1/valset \
#     --checkpoint /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/work_dir/DINOV2_B14_deepfake_c2/model_epoch_00000.pth \
#     --amp -e DINOV2_B14_deepfake_c2_002 --task_name DINOV2_B14_deepfake_c2_002 --device "cuda:0" --model_type dinov2_vitb14 --num_worker 8 --batch_size 32 \
#     --lr 1e-5 \
#      --evaluate_only

# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/phase1/testset_face_label.txt \
#     -val_img /nasdata2/private/lzhao/workspace/kaggle/DeepfakeDatasets/dataset \
#     --checkpoint /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/work_dir/DINOV2_B14_deepfake_c2_002/model_epoch_00000.pth \
#     --amp -e DINOV2_B14_deepfake_c2_002 --task_name DINOV2_B14_deepfake_c2_002 --device "cuda:0" --model_type dinov2_vitb14 --num_worker 8 --batch_size 32 \
#     --lr 1e-5 \
#     --evaluate_only

python train.py \
    -train /mnt/data/phase1/trainset_label.txt \
    -train_img /mnt/data/phase1/trainset \
    -val /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/image/phase2/testset1_seen_nolabel.txt \
    -val_img /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/image/phase2/testset1_seen \
    --checkpoint /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/work_dir/EfficientNet_B0_NPR_GRAD_deepfake_c2_002/model_best.pth \
    --amp -e EfficientNet_B0_NPR_GRAD_deepfake_c2_002_test --task_name EfficientNet_B0_NPR_GRAD_deepfake_c2_002_test --device "cuda:0" --model_type efficientnet_b0_npr_grad --num_worker 8 --batch_size 32 \
    --lr 1e-5 \
    --evaluate_only
    
# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /mnt/data/phase1/valset_label.txt \
#     -val_img /mnt/data/phase1/valset \
#     -e EfficientNet_B2_DS_deepfake_c2 \
#     --task_name EfficientNet_B2_DS_deepfake_c2 \
#     --device cuda:1 \
#     --model_type efficientnet_b2_ds \
#     --num_worker 8 \
#     --batch_size 64 \
#     --amp


# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /mnt/data/phase1/valset_label.txt \
#     -val_img /mnt/data/phase1/valset \
#     --amp -e EfficientNet_B0_NPR_GRAD_deepfake_c2_wo_smooth --task_name EfficientNet_B0_NPR_GRAD_deepfake_c2_wo_smooth --device "cuda:1" --model_type efficientnet_b0_npr_grad --num_worker 4 --batch_size 128


# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /mnt/data/phase1/valset_label.txt \
#     -val_img /mnt/data/phase1/valset \
#     --amp -e EfficientNet_B2_NPR_deepfake_c2 --task_name EfficientNet_B2_NPR_deepfake_c2 --device "cuda:0" --model_type efficientnet_b2_npr --num_worker 16 --batch_size 128

# python train.py --amp -e Resnet_101_NPR_deepfake_c2 --model_type resnet101
# python train.py --amp -e Resnet_101_NPR_deepfake_c2 --model_type resnet101