# Results Section

This repository is cleaned for GitHub upload. Actual generated outputs are not included because they are usually large and machine-specific.

## Available Result/Evaluation Logic

The experiment scripts contain evaluation code for:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Classification report
- Confusion matrix
- ROC/AUC curve, where applicable
- Training and validation loss/accuracy curves
- Fold-wise model checkpoint saving
- Fold history JSON saving

## Expected Generated Result Files After Running

The scripts are configured to save generated files under:

```text
/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE/
```

Expected files/folders include:

```text
clean_train_val_split.csv
clean_final_test_split.csv
fold_splits/
├── fold_1_train.csv
├── fold_1_val.csv
├── fold_1_config.json
├── ...
└── fold_5_config.json

<Model_Name>/
├── <Model_Name>_fold_1.pt
├── <Model_Name>_fold_2.pt
├── <Model_Name>_fold_3.pt
├── <Model_Name>_fold_4.pt
└── <Model_Name>_fold_5.pt

<Model_Name>_histories_clean.json
<Model_Name>_histories.json
final_test_eval_results.json
*_evaluation_summary.csv
```

## Recommended README Result Table

Use this table after running all experiments and filling the real values.

| Model / Experiment | Training Strategy | Accuracy | Precision | Recall | F1-score | Notes |
|---|---|---:|---:|---:|---:|---|
| DenseNet201 | Feature Extraction | TBD | TBD | TBD | TBD | 5-fold CV |
| DenseNet201 | Full Fine-Tuning | TBD | TBD | TBD | TBD | From FE checkpoint |
| DenseNet201 | Partial Fine-Tuning | TBD | TBD | TBD | TBD | True partial FT |
| ConvNeXt Small | Feature Extraction | TBD | TBD | TBD | TBD | 5-fold CV |
| ConvNeXt Small | Full Fine-Tuning | TBD | TBD | TBD | TBD | From FE checkpoint |
| DeiT Small | Feature Extraction | TBD | TBD | TBD | TBD | 5-fold CV |
| Swin Tiny | Full Fine-Tuning | TBD | TBD | TBD | TBD | From FE checkpoint |

## GitHub Note

Do not upload large `.pt`, `.pth`, `.ckpt`, raw datasets, or generated output folders unless required. Keep them in Google Drive or attach a small summary table/image only.
