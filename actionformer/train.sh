# python ./train.py  ./configs/AIC_2023.yaml \
#                     --output tmpp \
#                     --print-freq 1

# ======== MVIT =============
export PYTHONPATH="$PWD" && CUDA_VISIBLE_DEVICES=0 python ./train_all.py  ./configs/AIC_2023_MVit/AIC_2023_MVit.yaml \
                                                        --output concat_3view_2model \
                                                        --print-freq 5

# export PYTHONPATH="$PWD" && CUDA_VISIBLE_DEVICES=0 python ./train_single_view.py  ./configs/AIC_2023_old_feature.yaml \
#                                                         --output old_feature \
#                                                         --print-freq 10 \
#                                                         --num_folds 1 --view_idx 0                                                        