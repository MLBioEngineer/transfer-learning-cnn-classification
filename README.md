# Deep Learning Image Classification Portfolio

This repository contains the submitted deep-learning image-classification code notebooks prepared for GitHub upload and supervisor review. The experiments use PyTorch and `timm` models with cross-validation, training/evaluation loops, performance reports, confusion matrices, and training-curve visualizations.

## Included Experiments

| No. | Experiment | Notebook | Script |
|---:|---|---|---|
| 1 | DenseNet201 Feature Extraction | `notebooks/01_densenet201_feature_extraction.ipynb` | `scripts/01_densenet201_feature_extraction.py` |
| 2 | DenseNet201 Full Fine-Tuning | `notebooks/02_densenet201_full_fine_tuning.ipynb` | `scripts/02_densenet201_full_fine_tuning.py` |
| 3 | DenseNet201 Partial Fine-Tuning | `notebooks/03_densenet201_partial_fine_tuning.ipynb` | `scripts/03_densenet201_partial_fine_tuning.py` |
| 4 | ConvNeXt Small Feature Extraction | `notebooks/04_convnext_small_feature_extraction.ipynb` | `scripts/04_convnext_small_feature_extraction.py` |
| 5 | ConvNeXt Small Full Fine-Tuning | `notebooks/05_convnext_small_full_fine_tuning.ipynb` | `scripts/05_convnext_small_full_fine_tuning.py` |
| 6 | DeiT Small Feature Extraction | `notebooks/06_deit_small_feature_extraction.ipynb` | `scripts/06_deit_small_feature_extraction.py` |
| 7 | Swin Tiny Full Fine-Tuning From Feature Extraction | `notebooks/07_swin_tiny_full_fine_tuning_from_fe.ipynb` | `scripts/07_swin_tiny_full_fine_tuning_from_fe.py` |

## Repository Structure

```text
.
├── notebooks/                 # Clean GitHub-ready Jupyter notebooks
├── scripts/                   # Python scripts extracted from notebooks
├── docs/                      # Run order, GitHub upload guide, and file manifest
├── requirements.txt           # Python dependencies
├── .gitignore                 # Recommended GitHub ignore rules
└── README.md                  # Project overview
```

## How to Run

### Option 1: Google Colab

1. Open any notebook from the `notebooks/` folder in Google Colab.
2. Run the installation cell if present, for example `!pip install -q timm`.
3. Mount Google Drive when requested by the notebook.
4. Update the dataset, output, and checkpoint paths according to your own Google Drive folder.
5. Run cells sequentially.

### Option 2: Local Computer

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open the required notebook from the `notebooks/` folder.

## Dataset and Checkpoints

The dataset, trained model weights, and generated outputs are intentionally not included in this zip file because they are usually large and machine-specific. Place your dataset under a local or Google Drive folder and update the path variables inside the notebooks before running.

Recommended local layout:

```text
data/
├── train/
├── test/
└── labels.csv

outputs/
checkpoints/
results/
```

## Notes for GitHub Upload

- Upload the full extracted folder to GitHub.
- Keep `data/`, `outputs/`, `checkpoints/`, model weights, and large result files out of GitHub unless your supervisor specifically asks for them.
- The `.gitignore` file already excludes common large ML artifacts such as `.pt`, `.pth`, `.ckpt`, `.zip`, and output folders.
- The notebooks in this portfolio are cleaned for GitHub: cell outputs were removed to reduce file size and improve readability.

## Main Technologies

- Python
- PyTorch
- Torchvision
- timm
- scikit-learn
- pandas / NumPy
- Matplotlib / Seaborn
- Jupyter Notebook / Google Colab
