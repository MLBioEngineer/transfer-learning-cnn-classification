# Suggested Run Order

Use this order when reproducing the experiments or explaining the workflow to a supervisor.

1. `01_densenet201_feature_extraction.ipynb`  
   Establishes the DenseNet201 feature-extraction baseline.

2. `02_densenet201_full_fine_tuning.ipynb`  
   Performs full fine-tuning using the DenseNet201 feature-extraction setup/checkpoints.

3. `03_densenet201_partial_fine_tuning.ipynb`  
   Performs selective partial fine-tuning for DenseNet201.

4. `04_convnext_small_feature_extraction.ipynb`  
   Runs ConvNeXt Small as a feature extractor.

5. `05_convnext_small_full_fine_tuning.ipynb`  
   Performs full fine-tuning for ConvNeXt Small from feature-extraction weights.

6. `06_deit_small_feature_extraction.ipynb`  
   Runs the DeiT Small transformer-based feature-extraction experiment.

7. `07_swin_tiny_full_fine_tuning_from_fe.ipynb`  
   Runs the Swin Tiny workflow with feature-extraction checkpoint loading and full fine-tuning.

Before running any notebook, update dataset paths, output directories, and checkpoint locations according to your machine or Google Drive structure.
