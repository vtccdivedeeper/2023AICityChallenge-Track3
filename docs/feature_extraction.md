# Action Recognition (Feature extraction)
We adopt two action recognition backbones, namely, X3D model and MVitV2 model.

# Data preparation
- Split train - validation
```bash
cd /data_processing
python split_train_val.py --label_df_path /data/SetA1/processed_label_csv \
                            --output_path /path/to/output
```

# Train X3D
```bash
cd /PySlowFast-X3D
python tools/run_net.py --cfg configs/AICity2023/X3D_L.yaml
```
**Note**: Please modify the variables in the file config .yaml as follows:
- DATA.PATH_TO_DATA_DIR: path to dataset clip inputs, e.g, data/
- OUTPUT_DIR: path to output

# Train MViTv2
quanvm

# Feature extraction
```bash
# For X3D model:
cd /PySlowFast-X3D
python tools/extract_feature.py --cfg configs/AICity2023/X3D-L_extract_feature.yaml

# For MViTv2 model:
cd /PySlowFast-X3D
python tools/extract_feature.py --cfg configs/AICity2023/MViTv2_extract_feature.yaml
```

**Note**: Please modify the variables in the file config .yaml as follows:
- DATA.PATH_TO_DATA_DIR: path to dataset input, e.g., /data/SetA2.
- DATA.PATH_EXTRACT: path to feature output, e.g., /data/featureA2.
- WEIGHT: path to checkpoints' model.