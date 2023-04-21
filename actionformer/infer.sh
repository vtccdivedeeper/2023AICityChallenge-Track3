# ========= MVIT ===============
CUDA_VISIBLE_DEVICES=2 python ./inference_all.py ./configs/AIC_2023_X3D_MViTv2.yaml \
                 ./ckpt_25_03/AIC_2023_MVit_concat_3view_2model \
                 -epoch -1 \
                 --num_folds 5 \
                 --output_dir ./results/test_20_04/v2 \
                 --print-freq 10

# ========= X3D ==============
# CUDA_VISIBLE_DEVICES=1 python ./inference_all.py ./configs/AIC_2023_X3D/AIC_2023_infer.yaml \
#                  ./ckpt/AIC_2023_20_03_concat_3view \
#                  -epoch -1 \
#                  --num_folds 5 \
#                  --output_dir ./results/X3D/20_03_concat_3view_v7 \
#                  --print-freq 10

# ========= X3D ==============
# CUDA_VISIBLE_DEVICES=1 python ./inference_all_seperate_fold_feature.py ./configs/AIC_2023_X3D/AIC_2023_infer.yaml \
#                  ./ckpt/AIC_2023_20_03_concat_3view \
#                  -epoch 55 \
#                  --num_folds 5 \
#                  --output_dir ./results/X3D/21_03_5featuresA2_v2 \
#                  --print-freq 10