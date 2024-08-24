# export CUDA_VISIBLE_DEVICES=0,1
# python train.py \
#     --amp \
#     --device cuda:1 \
#     --evaluate_only \
#     -e Resnet_50_NPR_deepfake_c2 \
#     --model_type resnet50 \
#     --checkpoint work_dir/classify/model_best.pth

export CUDA_VISIBLE_DEVICES=0,1

# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /mnt/data/phase1/valset_label.txt \
#     -val_img /mnt/data/phase1/valset \
#     --amp -e EfficientNet_B2_NPR_GRAD_deepfake_c2 \
#     --task_name EfficientNet_B2_NPR_GRAD_deepfake_c2 \
#     --device "cuda:0" \
#     --evaluate_only \
#     --model_type efficientnet_b2_npr_grad \
#     --num_worker 8 \
#     --batch_size 128 \
#     --checkpoint work_dir/EfficientNet_B2_NPR_GRAD_deepfake_c2/model_best.pth

# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/phase1/testset_face_label.txt \
#     -val_img /nasdata2/private/lzhao/workspace/kaggle/DeepfakeDatasets/dataset \
#     --amp -e EfficientNet_B2_NPR_GRAD_deepfake_c2_testset \
#     --task_name EfficientNet_B2_NPR_GRAD_deepfake_c2_testset \
#     --device "cuda:0" \
#     --evaluate_only \
#     --model_type efficientnet_b2_npr_grad \
#     --num_worker 8 \
#     --batch_size 128 \
#     --checkpoint work_dir/EfficientNet_B2_NPR_GRAD_deepfake_c2/model_best.pth

# python train.py \
#     -train /mnt/data/phase1/trainset_label.txt \
#     -train_img /mnt/data/phase1/trainset \
#     -val /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/phase1/testset_face_label.txt \
#     -val_img /nasdata2/private/lzhao/workspace/kaggle/DeepfakeDatasets/dataset \
#     --amp -e EfficientNet_B2_NPR_GRAD_deepfake_c2_testset \
#     --task_name EfficientNet_B2_NPR_GRAD_deepfake_c2_testset \
#     --device "cuda:1" \
#     --evaluate_only \
#     --model_type efficientnet_b2_npr_grad \
#     --num_worker 2 \
#     --batch_size 4 \
#     --checkpoint work_dir/EfficientNet_B2_NPR_GRAD_deepfake_c2/model_best.pth


python train.py \
    -train /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/image/phase1/trainset_label.txt \
    -train_img /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/image/phase1/trainset \
    -val /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/image/phase2/testset1_seen_nolabel.txt \
    -val_img /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/image/phase2/testset1_seen \
    --checkpoint /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/work_dir/EfficientNet_B0_NPR_GRAD_deepfake_c2/model_best.pth \
    --amp -e EfficientNet_B0_NPR_GRAD_deepfake_c2_002_test --task_name EfficientNet_B0_NPR_GRAD_deepfake_c2_002_test --device "cuda:0" --model_type efficientnet_b0_npr_grad --num_worker 8 --batch_size 32 \
    --lr 1e-5 \
    --evaluate_only

# python train.py \
#     --amp \
#     -val /nasdata2/private/zwlu/classify/Kaggle/deepfake_adv/data/phase1/testset_label.txt\
#     -val_img /nasdata2/private/lzhao/workspace/kaggle/DeepfakeDatasets/dataset \
#     --device cuda:0 \
#     --evaluate_only \
#     -e Resnet_50_NPR_deepfake_c2 \
#     --model_type resnet50 \
#     --checkpoint work_dir/classify/model_best.pth