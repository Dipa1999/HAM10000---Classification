# HAM10000 Skin Lesion Classification

> Multi-model deep learning pipeline for dermoscopic image classification on the HAM10000 dataset, featuring custom CNNs, a DINOv2-based Vision Transformer, EfficientNetB3, and an interactive Streamlit web application.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Training Details](#training-details)
- [Results](#results)
- [Web Application](#web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

This project tackles the task of automated skin lesion classification using the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) dataset. Five deep learning models are trained and evaluated, ranging from a baseline custom CNN to a pretrained Vision Transformer, culminating in a weighted ensemble. An interactive Streamlit app allows inference on new images with per-class probability visualisation.

The full pipeline covers:

1. **Exploratory Data Analysis (EDA)** and preprocessing (Notebook `01_EDA_Preprocessing_DataLoaders.ipynb`)
2. **Model training, evaluation and Grad-CAM visualisation** (Notebook `02_Model_Training_Evaluation.ipynb`)
3. **Interactive inference web app** (`app.py`)

---

## Dataset

**HAM10000** (*Human Against Machine with 10000 training images*) is a large collection of multi-source dermatoscopic images of common pigmented skin lesions. It contains **10,015 images** across **7 diagnostic categories**:

| Code    | Full Name                        |
|---------|----------------------------------|
| `akiec` | Actinic Keratosis                |
| `bcc`   | Basal Cell Carcinoma             |
| `bkl`   | Benign Keratosis                 |
| `df`    | Dermatofibroma                   |
| `mel`   | Melanoma                         |
| `nv`    | Melanocytic Nevi                 |
| `vasc`  | Vascular Lesion                  |

The dataset is **highly imbalanced** (Melanocytic Nevi accounts for ~67% of samples). Class imbalance is addressed through `WeightedRandomSampler`, class-weighted focal loss, and aggressive augmentation.

**Download**: [Kaggle — Skin Lesion Analysis Toward Melanoma Detection](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)

### Splits

| Split | Samples |
|-------|---------|
| Train | 6 800   |
| Val   | 1 450   |
| Test  | 1 431   |

Splits are generated with `GroupShuffleSplit` to prevent patient-level data leakage (multiple images per patient are kept in the same split).

---

## Project Structure

```
.
├── data/                          # Raw downloaded data (not tracked by git)
├── processed/                     # Preprocessed splits, checkpoints, Grad-CAM outputs
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── best_hamcnn.pth
│   ├── best_enhanced_cnn.pth
│   ├── best_double_cnn.pth
│   ├── best_vit_dinov2.pth
│   ├── best_efficientnet_b3.pth
│   └── gradcam/
├── 01_EDA_Preprocessing_DataLoaders.ipynb
├── 02_Model_Training_Evaluation.ipynb
├── app.py                         # Streamlit web application
├── inference_utils.py             # Model loading and inference helpers
├── requirements.txt
└── README.md
```

---

## Models

Five models are defined, trained and evaluated, then combined into a weighted ensemble.

### 1. Baseline CNN (`HAMCNN`)

A custom convolutional network with 5 progressive stages `(32 → 64 → 128 → 256 → 384 channels)`, residual blocks, max-pooling downsampling, and a fully connected head with dropout.

### 2. Enhanced CNN

An improved version of the baseline with:
- **Feature Pyramid Network (FPN)** multi-scale feature fusion
- **Generalised Mean (GeM) pooling** instead of global average pooling
- **Stochastic Depth** regularisation (drop probability = 0.15)
- **SAM** (Sharpness-Aware Minimisation) optimiser

### 3. Double CNN

A dual-branch architecture sharing a common stem (`32 → 64` channels):
- **Classification branch**: continues to deeper stages with FPN, GeM, SAM
- **Reconstruction branch**: decodes stem features back to pixel space, trained with an auxiliary reconstruction loss (λ = 0.15) to encourage richer representations

### 4. Vision Transformer — DINOv2-Small (`ViT`)

`vit_small_patch14_dinov2` loaded from `timm` with ImageNet-21k pretrained weights:
- Backbone frozen for the first 5 epochs, then fine-tuned with differential learning rates (backbone: 2e-6, head: 5e-4)
- Custom 2-layer MLP head (hidden dim = 512)
- Gradient checkpointing available

### 5. EfficientNetB3

`efficientnet_b3` from `timm` pretrained on ImageNet-1k:
- Fine-tuned with differential LR (backbone: 5e-5, head: 5e-4)
- Custom head: `Dropout(0.4)` → `Linear(1536, 7)`
- Gradient accumulation over 2 steps

### Ensemble

Softmax outputs from all five models are combined with fixed weights optimised via **Differential Evolution** on the validation set:

| Model          | Weight |
|----------------|--------|
| HAMCNN         | 0.05   |
| Enhanced CNN   | 0.10   |
| Double CNN     | 0.10   |
| ViT-DINOv2     | 0.50   |
| EfficientNetB3 | 0.25   |

---

## Training Details

| Setting                  | Value                                    |
|--------------------------|------------------------------------------|
| Image size               | 224 × 224                                |
| Batch size               | 32                                       |
| Max epochs               | 50                                       |
| Early stopping patience  | 15 (metric: macro F1 on validation set)  |
| Loss function            | Focal Loss (γ = 1.5) + label smoothing (ε = 0.05) |
| Gradient clipping        | 2.0                                      |
| LR schedule              | Linear warmup (5 epochs) → Cosine annealing |
| Mixed precision (AMP)    | ✓ (all models)                           |
| Class balancing          | `WeightedRandomSampler` + class-weighted loss |
| Augmentation (train)     | Horizontal/Vertical flip, RandomRotate90, ElasticTransform, GridDistortion, OpticalDistortion, CLAHE, RandomShadow, ColorJitter, CoarseDropout |
| MixUp / CutMix           | α = 0.4 / p = 0.5                        |
| TTA (test time)          | 4 deterministic flips + 16 stochastic dropout passes |
| Hardware                 | NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)   |
| Framework                | PyTorch 2.5.1 + CUDA 12.1                |

---

## Results

Evaluation is performed on the held-out test set (n = 1 431) using the ensemble with TTA.

| Metric         | Value  |
|----------------|--------|
| Accuracy       | —      |
| Macro F1       | —      |
| AUC-ROC (macro)| —      |

> **Note**: Fill in the table above with your actual test-set metrics after running Notebook 02. Grad-CAM saliency maps are saved to `processed/gradcam/`.

---

## Web Application

The Streamlit app (`app.py`) provides:

- Image upload and interactive crop (`streamlit-cropper`)
- Ensemble inference with softmax probability bar chart (Plotly)
- Top-3 prediction display with confidence scores
- Grad-CAM overlay on the uploaded image

### Run the app

```bash
streamlit run app.py
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (recommended; CPU inference is supported but slow)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ham10000-skin-lesion-classification.git
cd ham10000-skin-lesion-classification

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Download the dataset

```bash
# Using the Kaggle CLI
kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
unzip skin-lesion-analysis-toward-melanoma-detection.zip -d data/
```

---

## Usage

### Step 1 — EDA and Preprocessing

Open and run `01_EDA_Preprocessing_DataLoaders.ipynb`. This will:

- Perform exploratory data analysis (class distribution, image statistics, duplicate detection)
- Standardise images and remove outliers
- Generate train/val/test CSV splits in `processed/`

### Step 2 — Model Training and Evaluation

Open and run `02_Model_Training_Evaluation.ipynb`. This will:

- Train all five models sequentially, saving the best checkpoint to `processed/`
- Evaluate each model and the ensemble on the test set
- Generate confusion matrices, Grad-CAM visualisations, and feature analysis plots

### Step 3 — Inference via Web App

```bash
streamlit run app.py
```

Upload a dermoscopic image and the ensemble will return predicted class probabilities.

---

## Limitations

- The model is trained exclusively on dermoscopic images; it will produce predictions for any input image (including photographs of non-skin subjects such as animals or objects) without raising an error. These predictions should be disregarded.
- This project is intended for **educational purposes only** and must not be used for clinical decision-making.
- Performance may degrade on images acquired with different dermoscopes or under different lighting conditions than those present in HAM10000.

---

## License

This project is released under the [MIT License](LICENSE).

The HAM10000 dataset is subject to its own license — please refer to the [original dataset page](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) for terms of use.
