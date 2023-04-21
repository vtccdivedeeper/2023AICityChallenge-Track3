# MVit
CUDA_VISIBLE_DEVICES=4 python ./eval.py ./configs/AIC_2023_MVit/AIC_2023_eval.yaml \
                                        ./ckpt_25_03_bs2/AIC_2023_MVit_concat_3view_2model/fold4 \
                                        -epoch -2 \
                                        --print-freq 5

# X3D
# CUDA_VISIBLE_DEVICES=4 python ./eval.py ./configs/AIC_2023_X3D/AIC_2023_eval.yaml \
#                                         ./ckpt/AIC_2023_15_03_bs1/fold4/Right_side_window \
#                                         -epoch -2 \
#                                         --print-freq 5