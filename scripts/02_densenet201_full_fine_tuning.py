"""
DenseNet201 Full Fine-Tuning

Auto-extracted from notebook: Full Fine Tunning .ipynb
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

    # same training hyperparameters for all individual models
    'lr': 1e-5,
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

    # individual CNN model setting - Updated for Full Fine-Tuning
    'freeze_backbone': False,
    'partial_finetune': False,
    'freeze_batchnorm': False,
    'num_unfrozen_layers_backbone': 0,
}

print("COMMON_CONFIG ready for Full Fine-Tuning.")

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

        # --- BATCHNORM HANDLING ---
        # In Full Fine-Tuning (freeze_batchnorm=False), we let BN update stats.
        # In Partial/Feature Extraction (freeze_batchnorm=True), we force BN to eval mode.
        if config.get('freeze_batchnorm'):
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        else:
            # Explicitly ensure BN is in training mode for full adaptation
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()

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
        print(f"Freezing backbone for {model_to_load} (Feature Extraction)")
        for name, param in model.named_parameters():
            if ('classifier' not in name) and ('head' not in name) and ('fc' not in name):
                param.requires_grad = False
    elif not config.get('freeze_backbone') and not config.get('partial_finetune'):
        print(f"Full Fine-Tuning mode: All layers for {model_to_load} are trainable.")
        for param in model.parameters():
            param.requires_grad = True

    return model

def train_single_model_same_config(model_spec, folds_info, base_output_dir, pretrained_model_name=None):
    model_to_load = model_spec['model_to_load']
    model_name = model_spec['model_name']

    model_save_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    histories = []

    print(f"\n==============================")
    print(f"Training model: {model_name}")
    print(f"Mode: {'Full Fine-Tuning' if not COMMON_CONFIG.get('freeze_backbone') else 'Transfer Learning'}")
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

        # 2. Load Pretrained Weights (if starting from a FE baseline)
        if pretrained_model_name:
            pretrained_path = os.path.join(base_output_dir, pretrained_model_name, f"{pretrained_model_name}_fold_{fold}.pt")
            if os.path.exists(pretrained_path):
                print(f"Loading checkpoint for fine-tuning: {pretrained_path}")
                model.load_state_dict(torch.load(pretrained_path, map_location=device))
            else:
                print(f"Warning: Pretrained weights not found at {pretrained_path}. Starting from ImageNet weights.")

        # 3. Handle Full Fine-Tuning overrides
        if not fold_config.get('freeze_backbone') and not fold_config.get('partial_finetune'):
            for param in model.parameters():
                param.requires_grad = True

        # 4. Partial Fine-Tuning logic (preserved)
        if fold_config.get('partial_finetune'):
            for param in model.parameters(): param.requires_grad = False
            for name, param in model.named_parameters():
                if 'features.denseblock4' in name or 'features.norm5' in name or 'classifier' in name or 'head' in name or 'fc' in name:
                    param.requires_grad = True

        model = model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=fold_config.get('label_smoothing', 0.0))

        # Optimizer (AdamW recommended for fine-tuning)
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=fold_config['lr'], weight_decay=fold_config.get('weight_decay', 1e-4))

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
    with open(os.path.join(base_output_dir, f"{model_name}_histories_clean.json"), "w") as f:
        json.dump(histories, f, indent=4)

    plot_training_curves(histories, model_name)
    return histories

# %% [cell 23]
import copy

print("Preparing to retrain ONLY Fold 1 for Feature Extraction...")

# Extract just the first fold's data from our existing folds_info
fold_1_info = copy.deepcopy(folds_info[0])

# Ensure the configuration is strictly set for Feature Extraction
# (in case it was modified by the partial fine-tuning cell)
fold_1_info['config']['freeze_backbone'] = True
fold_1_info['config']['partial_finetune'] = False
fold_1_info['config']['lr'] = .0001  # Original learning rate

model_spec_fe = {
    'model_to_load': 'densenet201',
    'model_name': 'DenseNet201_Feature_Extraction'
}

# Re-run the training function, but pass ONLY Fold 1 in the list
histories_fe_fold1 = train_single_model_same_config(
    model_spec=model_spec_fe,
    folds_info=[fold_1_info],  # Notice the brackets: it's a list containing just one fold
    base_output_dir=gdrive_base_output_dir,
    pretrained_model_name=None
)

print("\n✅ Fold 1 Retraining Complete! You can now re-run the cell that loads all 5 folds.")

# %% [cell 24]
import torch
import os
import json
import timm
import torch.nn as nn

model_name = 'DenseNet201_Feature_Extraction'
model_backbone = 'densenet201'
base_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

def build_timm_model(model_to_load, config):
    """Helper to rebuild the model architecture."""
    model = timm.create_model(
        model_to_load,
        pretrained=config.get('ImageNet', True),
        num_classes=config.get('num_classes', 4)
    )
    if config.get('freeze_backbone') and not config.get('partial_finetune'):
        for name, param in model.named_parameters():
            if ('classifier' not in name) and ('head' not in name) and ('fc' not in name):
                param.requires_grad = False
    return model

loaded_models = []

# Fix: If folds_info is missing from memory, try to reconstruct it from saved configs on Drive
if 'folds_info' not in globals():
    print("⚠️ 'folds_info' not found in memory. Attempting to recover configuration from Drive...")
    recovered_folds = []
    config_dir = os.path.join(base_dir, "fold_splits")

    for i in range(1, 6):
        config_path = os.path.join(config_dir, f'fold_{i}_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                recovered_folds.append({'fold': i, 'config': json.load(f)})

    if recovered_folds:
        folds_info = recovered_folds
        print(f"✅ Successfully recovered {len(folds_info)} fold configurations.")
    else:
        print("❌ Error: Could not find fold configurations on Drive. Please run the 'prepare_clean_folds' cell first.")

if 'folds_info' in globals():
    print(f"Loading all {len(folds_info)} folds for model: {model_name}...")

    for fold_to_load in range(1, len(folds_info) + 1):
        model_weights_path = os.path.join(base_dir, model_name, f'{model_name}_fold_{fold_to_load}.pt')

        print(f"\n--- Loading Fold {fold_to_load} ---")

        if os.path.exists(model_weights_path):
            fold_config = folds_info[fold_to_load - 1]['config']
            current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Now uses the locally defined function if the global one is missing
            loaded_model = build_timm_model(model_backbone, fold_config)

            try:
                loaded_model.load_state_dict(torch.load(model_weights_path, map_location=current_device))
                loaded_model = loaded_model.to(current_device)
                loaded_model.eval()
                loaded_models.append(loaded_model)
                print(f"Fold {fold_to_load} model successfully loaded!")
            except Exception as e:
                print(f"Error loading {model_weights_path}:\n{e}")
        else:
            print(f"Error: The file {model_weights_path} does not exist.")

    print(f"\nTotal models loaded successfully: {len(loaded_models)}")

# %% [cell 25]
import torch
import torchvision.transforms as transforms

# Helper to ensure mean/std are in list format for Normalize
def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

# Define transformation functions locally to prevent NameError on restart
def build_train_transform(config):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(
            size=(config['Resize_h'], config['Resize_w']),
            scale=(config['aug_random_resized_crop_scale_min'], config['aug_random_resized_crop_scale_max']),
            ratio=(0.95, 1.05)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=config['aug_random_rotation_degrees']),
        transforms.RandomAffine(
            degrees=0,
            translate=(config['aug_random_affine_translate_max'], config['aug_random_affine_translate_max']),
            scale=(config['aug_random_affine_scale_min'], config['aug_random_affine_scale_max']),
            shear=config['aug_random_affine_shear']
        ),
        transforms.ColorJitter(brightness=config['aug_color_jitter_brightness'], contrast=config['aug_color_jitter_contrast']),
        transforms.RandomAutocontrast(p=0.15),
        transforms.RandomAdjustSharpness(sharpness_factor=1.25, p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=config['aug_gaussian_blur_p']),
        transforms.ToTensor(),
        transforms.Normalize(mean=as_list(config['mean_overall']), std=as_list(config['std_overall'])),
        transforms.RandomErasing(p=config['aug_random_erasing_p'], scale=(0.01, 0.06), ratio=(0.3, 3.3), value='random')
    ])

def build_val_test_transform(config):
    return transforms.Compose([
        transforms.Resize((config['Resize_h'], config['Resize_w'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=as_list(config['mean_overall']), std=as_list(config['std_overall']))
    ])

model_spec_full = {
    'model_to_load': 'densenet201',
    'model_name': 'DenseNet201_Full_Fine_Tuning_From_FE'
}

# ============================================================
# CONFIG
# ============================================================
COMMON_CONFIG['freeze_backbone'] = False   # IMPORTANT: full fine-tuning
COMMON_CONFIG['partial_finetune'] = False  # disable partial
COMMON_CONFIG['freeze_batchnorm'] = False   # allow full adaptation
COMMON_CONFIG['lr'] = 1e-5
COMMON_CONFIG['weight_decay'] = 1e-4

# Apply same config to all folds
for fold_info in folds_info:
    fold_info['config']['freeze_backbone'] = False
    fold_info['config']['partial_finetune'] = False
    fold_info['config']['freeze_batchnorm'] = False
    fold_info['config']['lr'] = 1e-5
    fold_info['config']['weight_decay'] = 1e-4

# ============================================================
# TRAIN
# ============================================================
histories_densenet201_full = train_single_model_same_config(
    model_spec=model_spec_full,
    folds_info=folds_info,
    base_output_dir=gdrive_base_output_dir,
    pretrained_model_name='DenseNet201_Feature_Extraction'
)

# %% [cell 26]
import torch
import os
import numpy as np
from torch.utils.data import DataLoader

# Set random seed behavior to make results permanent/reproducible
SEED = 42
def set_eval_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_eval_seeds(SEED)

# 1. Configuration for Full Fine-Tuning evaluation
eval_model_name = 'DenseNet201_Full_Fine_Tuning_From_FE'
eval_backbone = 'densenet201'
base_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

# Re-load the models specifically from the Full Fine-Tuning directory
full_tuning_models = []
print(f"Loading Full Fine-Tuning models for Cross-Validation Ensemble...")

for fold_idx in range(1, 6):
    weights_path = os.path.join(base_dir, eval_model_name, f'{eval_model_name}_fold_{fold_idx}.pt')

    if os.path.exists(weights_path):
        fold_config = folds_info[fold_idx - 1]['config']
        model = timm.create_model(eval_backbone, pretrained=False, num_classes=fold_config['num_classes'])

        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model = model.to(device)
            model.eval()
            full_tuning_models.append(model)
            print(f"✅ Fold {fold_idx} weights loaded.")
        except Exception as e:
            print(f"❌ Error loading Fold {fold_idx}: {e}")
    else:
        print(f"⚠️ Warning: Weights for fold {fold_idx} not found.")

# 2. Prepare Test Data
if len(full_tuning_models) > 0:
    test_config = folds_info[0]['config']
    transform_test = build_val_test_transform(test_config)
    test_dataset = CustomImageDataset(clean_test_df, transform=transform_test, labels_map=test_config['labels_map'])

    # Ensuring deterministic dataloading
    g = torch.Generator()
    g.manual_seed(SEED)
    test_loader = DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=2, generator=g)

    # 3. Ensemble Evaluation (Averaging Probabilities across all folds)
    print(f"\nEvaluating Ensemble of {len(full_tuning_models)} Folds on Test Set...")

    all_fold_probs = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Ensemble Inference"):
            inputs = inputs.to(device)
            y_true.extend(labels.numpy())

            batch_probs = []
            for model in full_tuning_models:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())

            # Average probabilities for this batch across all models
            avg_batch_probs = np.mean(batch_probs, axis=0)
            all_fold_probs.extend(avg_batch_probs)

    all_fold_probs = np.array(all_fold_probs)
    y_pred = np.argmax(all_fold_probs, axis=1)
    y_true = np.array(y_true)

    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\n--- CV Ensemble (All Folds) Final Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=test_config['class_names'])}")

    # 4. Display Confusion Matrix for Ensemble
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=test_config['class_names'], yticklabels=test_config['class_names'])
    plt.title("Ensemble Confusion Matrix (All 5 Folds)")
    plt.show()

    if 'histories_densenet201_full' in globals():
        plot_training_curves(histories_densenet201_full, eval_model_name)
else:
    print("❌ No models loaded for evaluation.")

# %% [cell 27]
import torch
import os
import timm

# 1. Configuration
eval_model_name = 'DenseNet201_Full_Fine_Tuning_From_FE'
eval_backbone = 'densenet201'
base_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
final_save_path = os.path.join(base_dir, 'final_full_finetuned_model.pt')

full_tuning_models = []
print(f"Loading 5-fold models for: {eval_model_name}...")

# 2. Load all 5 folds
for fold_idx in range(1, 6):
    weights_path = os.path.join(base_dir, eval_model_name, f'{eval_model_name}_fold_{fold_idx}.pt')

    if os.path.exists(weights_path):
        # Use config from folds_info if available, else default to 4 classes
        num_classes_val = folds_info[fold_idx - 1]['config']['num_classes'] if 'folds_info' in globals() else 4
        model = timm.create_model(eval_backbone, pretrained=False, num_classes=num_classes_val)

        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model = model.to(device)
            model.eval()
            full_tuning_models.append(model)
            print(f" Fold {fold_idx} loaded.")
        except Exception as e:
            print(f" Error loading Fold {fold_idx}: {e}")
    else:
        print(f" Warning: Weights for fold {fold_idx} not found.")

# 3. Save the model
if len(full_tuning_models) > 0:
    # Saving the first model in the ensemble as the final representative model
    torch.save(full_tuning_models[0].state_dict(), final_save_path)
    print(f"\n Final model successfully saved to: {final_save_path}")
else:
    print("\n No models were loaded. Save operation aborted.")

# %% [cell 28]
import torch
import os

# Configuration
base_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
final_save_path = os.path.join(base_dir, 'final_full_finetuned_model.pt')

# Check if models are in memory
if 'full_tuning_models' in globals() and len(full_tuning_models) > 0:
    # Save the first model of the ensemble as the final checkpoint
    torch.save(full_tuning_models[0].state_dict(), final_save_path)
    print(f"✅ Final model successfully saved to: {final_save_path}")
else:
    print("❌ Error: 'full_tuning_models' not found in memory. Please run the evaluation/loading cell first.")
