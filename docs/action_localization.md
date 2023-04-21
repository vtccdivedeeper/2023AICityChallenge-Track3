# Action Localization with ActionFormer
We utilize ActionFormer for localizing actions. 
# 1. Data preparation
```bash
cd /data_processing
python create_data_json_action_former.py --split_csv_dir /path/to/split_csv \
                                                    --output_dir /path/to/json_format \
                                                    --name_folder_raw "SetA1" \
                                                    --name_folder_extract"Feature_A1" 
```
# 2. Training
```bash
cd /actionformer
python ./train_all.py   ./configs/AIC_2023_2model.yaml \
                        --output /path/to/output 
```
**Note**: Please modify the variables in the file config .yaml as follows:
- json_files: path to data json
- input_dim
- output_folder


# 3. Inference
Please follow steps in [README.md](../README.md)