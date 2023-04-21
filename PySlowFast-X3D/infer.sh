export PYTHONPATH="$PWD" && CUDA_VISIBLE_DEVICES=1 python tools/inference_ensemble_3views.py \
                        --cfg configs/AICity2023/X3D_L_reference.yaml \