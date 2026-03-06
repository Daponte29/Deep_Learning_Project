# CheXpert Deep Learning Project Experiment Plan

## 1. Project Overview
This project aims to develop and compare deep learning models for the automated interpretation of chest radiographs using the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/). The primary task is **multi-label classification** of 14 observations, focusing on the 5 competition-specific pathologies:
1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Pleural Effusion

## 2. Proposed Experiment: CNN vs. Transformer
The core experiment will compare the performance of a strong Convolutional Neural Network (CNN) baseline against a modern Vision Transformer (ViT) architecture.

### A. Baseline Model: DenseNet121
- **Why?**: DenseNet121 is the standard baseline established by the original CheXpert paper (Irvin et al., 2019). It captures fine-grained features through dense connections and is robust for medical imaging tasks.
- **Specifics**:
  - Pre-trained on ImageNet.
  - Global Average Pooling -> Linear Layer (14 outputs/5 outputs).
  - Input resolution: 320x320.

### B. Experimental Model: Vision Transformer (ViT-B/16)
- **Why?**: Transformers have shown state-of-the-art performance on natural images. Their global attention mechanism might capture long-range dependencies in X-rays (e.g., relating heart size to lung opacity) better than CNNs.
- **Hypothesis**: The ViT might struggle with the smaller effective dataset size (since we are starting with a subset) or show different generalization properties compared to the inductive bias of CNNs.
- **Specifics**:
  - Pre-trained on ImageNet-21k or ImageNet-1k.
  - Fine-tuned on CheXpert.

## 3. Methodology

### A. Data Preprocessing (Local & Cloud)
*   **Label Handling**: CheXpert contains `0` (negative), `1` (positive), and `-1` (uncertain).
    *   *Strategy*: **U-Ones** (treat uncertain as positive) or **U-Zeros** (treat uncertain as negative). We will start with **U-Ones** as it often yields high recall for critical pathologies.
*   **Image Augmentation**: Random rotation (+/- 10 degrees), slight zoom, and horizontal flip (careful with heart position, but valid for general lung opacity).
*   **Normalization**: Standard ImageNet mean/std.

### B. Training Pipeline
1.  **Local Dev (Small Scale)**: 
    *   Create a `sample.csv` with 1% of the data (~200 images).
    *   Confirm the pipeline runs end-to-end.
    *   **Sanity Check**: Overfit on a single batch (16 images) to achieve 0.0 loss.
2.  **Full Scale**:
    *   Train on the full dataset (or a large subset like 100k images).
    *   Evaluate AUROC on the validation set.

## 4. Evaluation Metrics
- **Primary**: Area Under the ROC Curve (AUROC) for each of the 5 pathologies.
- **Secondary**: F1-Score, Precision, Recall.

## 5. Potential Extensions
- **Ensembling**: Combine predictions from DenseNet and ViT.
- **Attention Maps**: Visualize where the ViT is "looking" using Attention Rollout vs. Grad-CAM for DenseNet.
