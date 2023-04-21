export PYTHONPATH="$PWD" && CUDA_VISIBLE_DEVICES=6 python tools/extract_feature.py \
                        --cfg configs/AICity2023/MViTv2_extract_feature.yaml \
# go to extract_feature.py and check dataset_name, use_ensemble_fold variables