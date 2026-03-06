# CheXpert Deep Learning Project: Experiment & Workflow

## 1. The Experiment: CNNs vs. Vision Transformers (ViT)

Our core objective is to compare the performance of established CNN architectures against modern Vision Transformers for multi-label classification of chest X-rays.

### Task
**Multi-label classification** of 5 specific pathologies:
1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Pleural Effusion

### Models
1.  **Baseline (CNN): DenseNet121**
    *   The "Gold Standard" for CheXpert.
    *   Known for feature reuse and parameter efficiency.
    *   *Hypothesis*: Strong performance with less data due to inductive bias (translation invariance).
2.  **Challenger (Transformer): ViT-B/16**
    *   State-of-the-art on natural images.
    *   Uses self-attention to model long-range dependencies (e.g., heart size relative to lung width).
    *   *Hypothesis*: May outperform CNNs if data scale is sufficient, but might struggle with overfitting on smaller subsets.

---

## 2. Technologies & Stack

*   **Deep Learning Framework**: [PyTorch](https://pytorch.org/) (Standard, flexible, widely used in research).
*   **Computer Vision**: [Torchvision](https://pytorch.org/vision/stable/index.html) (Pre-trained weights for DenseNet/ViT, transforms).
*   **Data Handling**: [Pandas](https://pandas.pydata.org/) (CSV manipulation), [Pillow](https://python-pillow.org/) (Image loading).
*   **Training Loop**: Custom PyTorch loop (for maximum control and learning experience) or Pytorch Lightning (optional later).
*   **Environment**: Anaconda / Python `venv`.

---

## 3. Local Development Workflow (Step-by-Step)

The CheXpert dataset is **400GB**, which is impossible to iterate on quickly. Follow this workflow:

### A. Setup Data (The "Tiny" Strategy)
Do not download the full dataset to your laptop.
1.  Download the **CheXpert-v1.0-small** (low res version, ~11GB) or just a few sample folders.
2.  **Resize further**: Use `src/data/preprocess.py` to create 224x224 versions of images. This drastically speeds up loading times.
3.  **Sample CSV**: Create a `sample.csv` containing only 100 rows. Use this for all code debugging.

### B. Run the Pipeline
1.  **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Debug run (1 epoch, tiny batch)**:
    ```bash
    python -m src.train --csv_path data/sample.csv --epochs 1 --batch_size 2
    ```

---

## 4. Deep Learning Tips & Debugging Guide

### A. The "Overfit Single Batch" Test
Before training on the full dataset, **always** try to overfit a single batch of data.
*   **Goal**: Drive Training Loss to roughly **0.0**.
*   **Why?**: If your model cannot memorize 16 images, it has a bug (e.g., frozen weights, broken forward pass, wrong loss function).
*   **How**:
    1.  Set `batch_size = 8`.
    2.  Take the first batch from the dataloader.
    3.  Loop over *just that batch* 100 times.
    4.  Verify loss drops dramatically.

### B. Checkpointing
Training takes hours/days. Always save your model state.
*   **Save Logic**: We implemented saving in `src/train.py`. It saves `best_model.pth` whenever validation AUROC improves.
*   **Resume**: If your computer crashes, load `model.load_state_dict(torch.load('path.pth'))` to continue.

### C. Debugging Dataloaders
The most common error source is data loading.
*   **Visual Check**: Use `matplotlib` to plot a batch of images *after* transforms but *before* the model. Ensure labels overlap correctly (if using segmentation) or look reasonable.
*   **Shape Check**: Print `images.shape` and `labels.shape` in the first loop.
    *   Expected: `[Batch, 3, 224, 224]` and `[Batch, 5]`.

### D. Hyperparameters to Tune
1.  **Learning Rate**: Start with `1e-4`. If loss explodes (NaN), lower to `1e-5`. If loss doesn't move, try `1e-3`.
2.  **Batch Size**: Maximize this to fit your GPU memory.
3.  **Image Size**: 224x224 is standard. 320x320 is better for pathology detection but strictly slower.

### E. Handling Class Imbalance
Medical datasets are imbalanced (e.g., very few "Fracture" cases).
*   **Pos_Weight**: In `BCEWithLogitsLoss`, use the `pos_weight` argument to weight positive classes higher.
*   **Sampling**: Use a `WeightedRandomSampler` to oversample rare classes during training.
