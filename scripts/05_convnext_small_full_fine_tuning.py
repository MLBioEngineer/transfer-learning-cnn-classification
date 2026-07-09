"""
ConvNeXt Small Full Fine-Tuning

Auto-extracted from notebook: convnext full fine tunnig  .ipynb
Notebook cell boundaries are preserved as comments.
Colab shell/magic commands are commented out for script readability.
"""


# %% [cell 1]
# !pip install -q timm

# %% [cell 2]
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# %% [cell 3]
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import os
import json
import copy
import random
import zipfile
import hashlib
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import timm

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 42
set_all_seeds(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% [cell 4]
import os
import zipfile

zip_file_path = '/content/gdrive/MyDrive/archive (2).zip'
extract_dir = '/content/unzipped_data'

# Check if the file exists before attempting to unzip
if os.path.exists(zip_file_path):
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Dataset extracted to: {extract_dir}")
else:
    print(f"Error: File not found at {zip_file_path}")
    print("Please ensure Google Drive is mounted and the file path is correct.")

# %% [cell 5]
base_data_dir = '/content/unzipped_data/Data'
image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

def find_images_and_labels_from_subdirs(root_dir, extensions):
    image_paths = []
    labels = []

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)

        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(extensions):
                    image_paths.append(os.path.join(class_dir, file))
                    labels.append(class_name)

    return image_paths, labels

train_paths, train_labels = find_images_and_labels_from_subdirs(
    os.path.join(base_data_dir, 'train'),
    image_extensions
)

valid_paths, valid_labels = find_images_and_labels_from_subdirs(
    os.path.join(base_data_dir, 'valid'),
    image_extensions
)

test_paths, test_labels = find_images_and_labels_from_subdirs(
    os.path.join(base_data_dir, 'test'),
    image_extensions
)

train_initial_df = pd.DataFrame({
    'filepath': train_paths,
    'label': train_labels,
    'original_split': 'train'
})

valid_initial_df = pd.DataFrame({
    'filepath': valid_paths,
    'label': valid_labels,
    'original_split': 'valid'
})

test_initial_df = pd.DataFrame({
    'filepath': test_paths,
    'label': test_labels,
    'original_split': 'test'
})

print("Original train:", len(train_initial_df))
print("Original valid:", len(valid_initial_df))
print("Original test :", len(test_initial_df))

# %% [cell 6]
label_mapping = {
    'normal':                                            'normal',
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib':       'adenocarcinoma',
    'adenocarcinoma':                                    'adenocarcinoma',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'squamous.cell.carcinoma',
    'squamous.cell.carcinoma':                          'squamous.cell.carcinoma',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa':    'large.cell.carcinoma',
    'large.cell.carcinoma':                             'large.cell.carcinoma',
}

all_df = pd.concat(
    [train_initial_df, valid_initial_df, test_initial_df],
    ignore_index=True
)

all_df['label'] = all_df['label'].map(label_mapping)
all_df = all_df.dropna(subset=['label']).reset_index(drop=True)

print("Total images before duplicate removal:", len(all_df))
print("\nClass distribution before duplicate removal:")
print(all_df['label'].value_counts())

# %% [cell 7]
def file_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()

all_df['file_hash'] = all_df['filepath'].apply(file_hash)

print("Total rows:", len(all_df))
print("Unique hashes:", all_df['file_hash'].nunique())
print("Exact duplicate rows:", len(all_df) - all_df['file_hash'].nunique())

# %% [cell 8]
conflict_hashes = (
    all_df.groupby('file_hash')['label']
    .nunique()
    .reset_index()
)

conflict_hashes = conflict_hashes[conflict_hashes['label'] > 1]

print("Conflicting duplicate hashes:", len(conflict_hashes))

if len(conflict_hashes) > 0:
    conflict_examples = all_df[all_df['file_hash'].isin(conflict_hashes['file_hash'])]
    display(conflict_examples.sort_values('file_hash').head(50))

# %% [cell 9]
clean_df = all_df.drop_duplicates(subset=['file_hash']).reset_index(drop=True)

print("Total images after duplicate removal:", len(clean_df))
print("Removed duplicates:", len(all_df) - len(clean_df))

print("\nClean class distribution:")
print(clean_df['label'].value_counts())

# %% [cell 10]
clean_train_val_df, clean_test_df = train_test_split(
    clean_df,
    test_size=0.20,
    stratify=clean_df['label'],
    random_state=SEED
)

clean_train_val_df = clean_train_val_df.reset_index(drop=True)
clean_test_df = clean_test_df.reset_index(drop=True)

print("Clean Train+Val:", len(clean_train_val_df))
print("Clean Final Test:", len(clean_test_df))

print("\nClean Train+Val distribution:")
print(clean_train_val_df['label'].value_counts())

print("\nClean Final Test distribution:")
print(clean_test_df['label'].value_counts())

# %% [cell 11]
def check_hash_overlap(df1, df2, name1, name2):
    overlap = set(df1['file_hash']) & set(df2['file_hash'])
    print(f"{name1} vs {name2}: {len(overlap)} duplicate hashes")

check_hash_overlap(clean_train_val_df, clean_test_df, "Clean TrainVal", "Clean Final Test")

# %% [cell 12]
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
os.makedirs(gdrive_base_output_dir, exist_ok=True)

clean_train_val_csv = os.path.join(gdrive_base_output_dir, "clean_train_val_split.csv")
clean_test_csv = os.path.join(gdrive_base_output_dir, "clean_final_test_split.csv")

clean_train_val_df.to_csv(clean_train_val_csv, index=False)
clean_test_df.to_csv(clean_test_csv, index=False)

print("Saved:", clean_train_val_csv)
print("Saved:", clean_test_csv)

# %% [cell 13]
class_names = sorted(clean_df['label'].unique().tolist())
num_classes = len(class_names)
labels_map = {label: i for i, label in enumerate(class_names)}

print("Class names:", class_names)
print("Num classes:", num_classes)
print("Labels map:", labels_map)

# %% [cell 14]
COMMON_CONFIG = {
    # general
    'input_ch': 3,
    'ImageNet': True,
    'num_folds': 5,
    'Resize_h': 224,
    'Resize_w': 224,
    'stop_criteria': 'loss',

    # hyperparameters for full fine-tuning
    'lr': 5e-6,                    # Lower learning rate for fine-tuning
    'optim_fc': 'AdamW',
    'weight_decay': 1e-4,
    'batch_size': 16,
    'n_epochs': 50,
    'max_epochs_stop': 30,
    'epochs_patience': 10,

    # class info
    'num_classes': num_classes,
    'class_names': class_names,
    'labels_map': labels_map,

    # augmentation
    'aug_random_resized_crop_scale_min': 0.88,
    'aug_random_resized_crop_scale_max': 1.00,
    'aug_random_rotation_degrees': 7,
    'aug_color_jitter_brightness': 0.08,
    'aug_color_jitter_contrast': 0.10,
    'aug_color_jitter_saturation': 0.0,
    'aug_color_jitter_hue': 0.0,
    'aug_random_affine_translate_max': 0.04,
    'aug_random_affine_scale_min': 0.96,
    'aug_random_affine_scale_max': 1.04,
    'aug_random_affine_shear': 3,
    'aug_random_vertical_flip_p': 0.0,
    'aug_random_grayscale_p': 0.0,
    'aug_gaussian_blur_p': 0.08,
    'aug_random_perspective_p': 0.0,
    'aug_random_erasing_p': 0.12,

    # regularization
    'label_smoothing': 0.05,

    # Full Fine-Tuning settings
    'freeze_backbone': False,      # Unfreeze all layers
    'partial_finetune': False,
    'freeze_batchnorm': False,     # Allow normalization layers to update
    'num_unfrozen_layers_backbone': 'all',
}

print("COMMON_CONFIG ready for ConvNeXt Full Fine-Tuning.")

# %% [cell 15]
def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

def build_train_transform(config):
    return transforms.Compose([
        transforms.Resize((256, 256)),

        transforms.RandomResizedCrop(
            size=(config['Resize_h'], config['Resize_w']),
            scale=(
                config['aug_random_resized_crop_scale_min'],
                config['aug_random_resized_crop_scale_max']
            ),
            ratio=(0.95, 1.05)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=config['aug_random_rotation_degrees']
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(
                config['aug_random_affine_translate_max'],
                config['aug_random_affine_translate_max']
            ),
            scale=(
                config['aug_random_affine_scale_min'],
                config['aug_random_affine_scale_max']
            ),
            shear=config['aug_random_affine_shear']
        ),

        transforms.ColorJitter(
            brightness=config['aug_color_jitter_brightness'],
            contrast=config['aug_color_jitter_contrast'],
            saturation=0.0,
            hue=0.0
        ),

        transforms.RandomAutocontrast(p=0.15),
        transforms.RandomAdjustSharpness(sharpness_factor=1.25, p=0.15),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=config['aug_gaussian_blur_p']),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=as_list(config['mean_overall']),
            std=as_list(config['std_overall'])
        ),

        transforms.RandomErasing(
            p=config['aug_random_erasing_p'],
            scale=(0.01, 0.06),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])

# %% [cell 16]
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, labels_map=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels_map = labels_map

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert('RGB')

        label_name = self.dataframe.iloc[idx]['label']
        label = self.labels_map[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label

# %% [cell 17]
def compute_mean_std_from_dataframe(dataframe):
    means, stds = [], []

    for img_path in tqdm(dataframe['filepath'], desc="Computing mean/std"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img) / 255.0

            means.append(img_np.mean(axis=(0, 1)))
            stds.append(img_np.std(axis=(0, 1)))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    mean_overall = np.mean(means, axis=0)
    std_overall = np.mean(stds, axis=0)

    print("Mean:", mean_overall)
    print("Std :", std_overall)

    return mean_overall, std_overall

# %% [cell 18]
def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

def build_train_transform(config):
    return transforms.Compose([
        transforms.Resize((256, 256)),

        transforms.RandomResizedCrop(
            size=(config['Resize_h'], config['Resize_w']),
            scale=(
                config['aug_random_resized_crop_scale_min'],
                config['aug_random_resized_crop_scale_max']
            ),
            ratio=(0.95, 1.05)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=config['aug_random_rotation_degrees']
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(
                config['aug_random_affine_translate_max'],
                config['aug_random_affine_translate_max']
            ),
            scale=(
                config['aug_random_affine_scale_min'],
                config['aug_random_affine_scale_max']
            ),
            shear=config['aug_random_affine_shear']
        ),

        transforms.ColorJitter(
            brightness=config['aug_color_jitter_brightness'],
            contrast=config['aug_color_jitter_contrast'],
            saturation=0.0,
            hue=0.0
        ),

        transforms.RandomAutocontrast(p=0.15),
        transforms.RandomAdjustSharpness(sharpness_factor=1.25, p=0.15),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=config['aug_gaussian_blur_p']),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=as_list(config['mean_overall']),
            std=as_list(config['std_overall'])
        ),

        transforms.RandomErasing(
            p=config['aug_random_erasing_p'],
            scale=(0.01, 0.06),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])

def build_val_test_transform(config):
    return transforms.Compose([
        transforms.Resize((config['Resize_h'], config['Resize_w'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=as_list(config['mean_overall']),
            std=as_list(config['std_overall'])
        )
    ])

print("Dataset and transforms ready.")

# %% [cell 19]
def make_json_serializable_config(config):
    cfg = {}

    for k, v in config.items():
        if isinstance(v, np.ndarray):
            cfg[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            cfg[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            cfg[k] = int(v)
        else:
            cfg[k] = v

    return cfg
def prepare_clean_folds(clean_train_val_df, base_config, output_dir):
    skf = StratifiedKFold(
        n_splits=base_config['num_folds'],
        shuffle=True,
        random_state=SEED
    )

    folds_info = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(clean_train_val_df['filepath'], clean_train_val_df['label'])
    ):
        fold_num = fold + 1

        print(f"\nPreparing Fold {fold_num}/{base_config['num_folds']}")

        train_fold_df = clean_train_val_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = clean_train_val_df.iloc[val_idx].reset_index(drop=True)

        hash_overlap = set(train_fold_df['file_hash']) & set(val_fold_df['file_hash'])
        print("Train-Val duplicate hash overlap:", len(hash_overlap))

        if len(hash_overlap) > 0:
            raise ValueError(f"Leakage found inside Fold {fold_num}")

        # Strict normalization: train fold only
        mean_fold, std_fold = compute_mean_std_from_dataframe(train_fold_df)

        fold_config = base_config.copy()
        fold_config['mean_overall'] = mean_fold
        fold_config['std_overall'] = std_fold

        # save fold split
        split_dir = os.path.join(output_dir, "fold_splits")
        os.makedirs(split_dir, exist_ok=True)

        train_fold_df.to_csv(
            os.path.join(split_dir, f"fold_{fold_num}_train.csv"),
            index=False
        )

        val_fold_df.to_csv(
            os.path.join(split_dir, f"fold_{fold_num}_val.csv"),
            index=False
        )

        # save fold config
        fold_config_json = make_json_serializable_config(fold_config)

        with open(os.path.join(split_dir, f"fold_{fold_num}_config.json"), "w") as f:
            json.dump(fold_config_json, f, indent=4)

        folds_info.append({
            'fold': fold_num,
            'train_df': train_fold_df,
            'val_df': val_fold_df,
            'config': fold_config
        })

    return folds_info
folds_info = prepare_clean_folds(
    clean_train_val_df=clean_train_val_df,
    base_config=COMMON_CONFIG,
    output_dir=gdrive_base_output_dir
)

print("All clean folds prepared.")

# %% [cell 20]
print(f"Number of folds: {COMMON_CONFIG['num_folds']}")

for i, fold_data in enumerate(folds_info):
    fold_num = fold_data['fold']
    train_df = fold_data['train_df']
    val_df = fold_data['val_df']

    print(f"\nFold {fold_num}:")
    print(f"  Number of training images: {len(train_df)}")
    print(f"  Number of validation images: {len(val_df)}")
    print(f"  Training class distribution:\n{train_df['label'].value_counts()}")
    print(f"  Validation class distribution:\n{val_df['label'].value_counts()}")

# %% [cell 21]
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, save_path=None):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }

    for epoch in range(config['n_epochs']):
        model.train()

        # --- BATCHNORM / LAYERNORM HANDLING ---
        # If freeze_batchnorm is True (Feature Extraction), force those layers to eval mode.
        # If False (Full Fine-Tuning), we let model.train() handle it naturally.
        if config.get('freeze_batchnorm', False):
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                    module.eval()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']} Train"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train

        model.eval()

        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']} Val"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        print(
            f"Epoch {epoch+1}/{config['n_epochs']} | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())

            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved: {save_path}")

        else:
            epochs_no_improve += 1
            print(f"No improvement: {epochs_no_improve}/{config['epochs_patience']}")

            if epochs_no_improve >= config['epochs_patience']:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, dataloader, device, class_names, title="Evaluation", show_plots=True):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=title):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\n--- {title} Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    if show_plots:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f"{title} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} AUC={roc_auc:.2f}")

        plt.plot([0, 1], [0, 1], '--', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title} ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs
    }

def plot_training_curves(histories, model_name):
    num_folds = len(histories)

    fig, axes = plt.subplots(num_folds, 2, figsize=(15, 5 * num_folds))

    if num_folds == 1:
        axes = [axes]

    fig.suptitle(f"Training and Validation Curves: {model_name}", fontsize=16)

    for i, history in enumerate(histories):
        axes[i][0].plot(history['train_loss'], label='Train Loss')
        axes[i][0].plot(history['val_loss'], label='Val Loss')
        axes[i][0].set_title(f"Fold {i+1} Loss")
        axes[i][0].set_xlabel("Epoch")
        axes[i][0].set_ylabel("Loss")
        axes[i][0].legend()
        axes[i][0].grid(True)

        axes[i][1].plot(history['train_acc'], label='Train Accuracy')
        axes[i][1].plot(history['val_acc'], label='Val Accuracy')
        axes[i][1].set_title(f"Fold {i+1} Accuracy")
        axes[i][1].set_xlabel("Epoch")
        axes[i][1].set_ylabel("Accuracy")
        axes[i][1].legend()
        axes[i][1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %% [cell 22]
def build_timm_model(model_to_load, config):
    model = timm.create_model(
        model_to_load,
        pretrained=config['ImageNet'],
        num_classes=config['num_classes']
    )

    # Handle Freezing Logic
    if config.get('freeze_backbone') and not config.get('partial_finetune'):
        print(f"Freezing backbone for {model_to_load} (Feature Extraction Mode)")
        for name, param in model.named_parameters():
            # Identify head/classifier - standard timm naming: 'head', 'fc', 'classifier'
            if not any(x in name for x in ['classifier', 'head', 'fc']):
                param.requires_grad = False
            else:
                param.requires_grad = True

    elif not config.get('freeze_backbone') and not config.get('partial_finetune'):
        print(f"Full Fine-Tuning mode: All layers for {model_to_load} are trainable.")
        for param in model.parameters():
            param.requires_grad = True

    # Partial fine-tuning is handled as a separate override if specifically requested
    return model

def train_single_model_same_config(model_spec, folds_info, base_output_dir, pretrained_model_name=None):
    model_to_load = model_spec['model_to_load']
    model_name = model_spec['model_name']

    model_save_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    histories = []

    print(f"\n==============================")
    print(f"Training model: {model_name}")
    print(f"Mode: {'Feature Extraction' if COMMON_CONFIG.get('freeze_backbone') else 'Full Fine-Tuning'}")
    print(f"==============================")

    for fold_info in folds_info:
        fold = fold_info['fold']
        train_fold_df = fold_info['train_df']
        val_fold_df = fold_info['val_df']

        fold_config = fold_info['config'].copy()
        fold_config['model_to_load'] = model_to_load
        fold_config['model_name'] = model_name
        fold_config['save_path'] = model_save_dir

        print(f"\nStarting Fold {fold}/{fold_config['num_folds']}")

        transform_train = build_train_transform(fold_config)
        transform_val = build_val_test_transform(fold_config)

        train_loader = DataLoader(CustomImageDataset(train_fold_df, transform=transform_train, labels_map=fold_config['labels_map']),
                                batch_size=fold_config['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(CustomImageDataset(val_fold_df, transform=transform_val, labels_map=fold_config['labels_map']),
                              batch_size=fold_config['batch_size'], shuffle=False, num_workers=2)

        # 1. Build Model
        model = build_timm_model(model_to_load, fold_config)

        # 2. Load Pretrained Weights (if starting fine-tuning from a FE checkpoint)
        if pretrained_model_name:
            pretrained_path = os.path.join(base_output_dir, pretrained_model_name, f"{pretrained_model_name}_fold_{fold}.pt")
            if os.path.exists(pretrained_path):
                print(f"Loading weights from: {pretrained_path}")
                model.load_state_dict(torch.load(pretrained_path, map_location=device))
            else:
                print(f"Warning: Pretrained weights not found at {pretrained_path}. Using ImageNet/Default initialization.")

        # 3. Handle Partial Finetuning overrides if specific block freezing is needed
        if fold_config.get('partial_finetune'):
            print("Applying Partial Fine-Tuning overrides...")
            for param in model.parameters(): param.requires_grad = False
            for name, param in model.named_parameters():
                # Example blocks for ConvNeXt/ResNet types
                if any(x in name for x in ['stages.3', 'norm_pre', 'head', 'classifier', 'fc']):
                    param.requires_grad = True

        model = model.to(device)

        # Only pass parameters with requires_grad=True to the optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=fold_config['lr'], weight_decay=fold_config.get('weight_decay', 1e-4))

        criterion = nn.CrossEntropyLoss(label_smoothing=fold_config.get('label_smoothing', 0.0))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=fold_config['epochs_patience'] // 2)

        fold_save_path = os.path.join(model_save_dir, f"{model_name}_fold_{fold}.pt")

        model_trained, history = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            config=fold_config, device=device, save_path=fold_save_path
        )

        histories.append(history)
        evaluate_model(model_trained, val_loader, device, fold_config['class_names'], title=f"Fold {fold} Val")

    # Save aggregated histories
    history_path = os.path.join(base_output_dir, f"{model_name}_histories.json")
    with open(history_path, "w") as f:
        json.dump(histories, f, indent=4)

    plot_training_curves(histories, model_name)
    return histories

# %% [cell 23]
loaded_models = []
model_name = "ConvNeXt_Small_Feature_Extraction"
model_to_load = "convnext_small"

print(f"Loading models from: {gdrive_base_output_dir}/{model_name}")

for fold_info in folds_info:
    fold = fold_info['fold']
    fold_config = fold_info['config']

    # 1. Re-build the architecture
    model = build_timm_model(model_to_load, fold_config)

    # 2. Path to the saved weights
    weights_path = os.path.join(gdrive_base_output_dir, model_name, f"{model_name}_fold_{fold}.pt")

    # 3. Load the weights
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        loaded_models.append(model)
        print(f"Successfully loaded Fold {fold} weights.")
    else:
        print(f"Error: Weights for Fold {fold} not found at {weights_path}")

print(f"\nTotal models loaded: {len(loaded_models)}")

# Save the list of loaded models (Ensemble)
ensemble_save_path = os.path.join(gdrive_base_output_dir, model_name, f"{model_name}_ensemble_list.pt")
torch.save(loaded_models, ensemble_save_path)
print(f"Ensemble models list saved to: {ensemble_save_path}")

# %% [cell 24]
# ============================================================
# ConvNeXt-Small Full Fine-Tuning from Feature Extraction
# ============================================================

model_spec_convnext_full = {
    'model_to_load': 'convnext_small',
    'model_name': 'ConvNeXt_Small_Full_Fine_Tuning_From_FE'
}

# ============================================================
# CONFIG FOR FULL FINE-TUNING
# ============================================================

COMMON_CONFIG['freeze_backbone'] = False      # full fine-tuning: all layers trainable
COMMON_CONFIG['partial_finetune'] = False     # disable partial fine-tuning
COMMON_CONFIG['freeze_batchnorm'] = False     # ConvNeXt mostly uses LayerNorm, keep this False
COMMON_CONFIG['lr'] = 5e-6                    # low LR because FE already very strong
COMMON_CONFIG['weight_decay'] = 1e-4

# Update every fold config also
for fold_info in folds_info:
    fold_info['config']['freeze_backbone'] = False
    fold_info['config']['partial_finetune'] = False
    fold_info['config']['freeze_batchnorm'] = False
    fold_info['config']['lr'] = 5e-6
    fold_info['config']['weight_decay'] = 1e-4

# ============================================================
# TRAIN FULL FINE-TUNING
# ============================================================

histories_convnext_full = train_single_model_same_config(
    model_spec=model_spec_convnext_full,
    folds_info=folds_info,
    base_output_dir=gdrive_base_output_dir,
    pretrained_model_name='ConvNeXt_Small_Feature_Extraction'
)

# %% [cell 25]
loaded_models_full = []
model_name_full = "ConvNeXt_Small_Full_Fine_Tuning_From_FE"
model_to_load = "convnext_small"

print(f"Loading Full Fine-Tuning models from: {gdrive_base_output_dir}/{model_name_full}")

for fold_info in folds_info:
    fold = fold_info['fold']
    fold_config = fold_info['config']

    # 1. Re-build the architecture
    model = build_timm_model(model_to_load, fold_config)

    # 2. Path to the saved weights
    weights_path = os.path.join(gdrive_base_output_dir, model_name_full, f"{model_name_full}_fold_{fold}.pt")

    # 3. Load the weights
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        loaded_models_full.append(model)
        print(f"Successfully loaded Fold {fold} weights.")
    else:
        print(f"Error: Weights for Fold {fold} not found at {weights_path}")

print(f"\nTotal Full FT models loaded: {len(loaded_models_full)}")

# Save the list of loaded models (Ensemble)
ensemble_full_save_path = os.path.join(gdrive_base_output_dir, model_name_full, f"{model_name_full}_ensemble_list.pt")
torch.save(loaded_models_full, ensemble_full_save_path)
print(f"Ensemble models list saved to: {ensemble_full_save_path}")

# %% [cell 26]
print("Checking for internal leakage in each fold (Train vs Val overlap):")
for fold_data in folds_info:
    fold_num = fold_data['fold']
    train_hashes = set(fold_data['train_df']['file_hash'])
    val_hashes = set(fold_data['val_df']['file_hash'])

    overlap = train_hashes.intersection(val_hashes)
    print(f"Fold {fold_num}: {len(overlap)} overlapping hashes")

    if len(overlap) > 0:
        print(f"--- ALERT: Leakage found in Fold {fold_num}! ---")
        display(fold_data['train_df'][fold_data['train_df']['file_hash'].isin(overlap)].head())

print("\nChecking for leakage between Training/Val and the Final Test set:")
test_hashes = set(clean_test_df['file_hash'])
train_val_hashes = set(clean_train_val_df['file_hash'])

overlap_test = train_val_hashes.intersection(test_hashes)
print(f"Overlap between Train/Val and Final Test: {len(overlap_test)} hashes")
