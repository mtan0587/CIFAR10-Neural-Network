# ELEC5304 — Project 2: Image Classification Networks

**University of Sydney | Image Processing (ELEC5304) | Semester 1, 2026**

This project implements and evaluates a compact residual CNN for image classification on the CIFAR-10 dataset, trained from scratch with a custom training pipeline built in PyTorch.

---

## Model — ResNet9

A 9-layer residual network designed to reach high accuracy quickly, without needing a large compute budget.

```
Input (3 × 32 × 32)
  → stem   Conv 3→64
  → proj   Conv 64→64  (1×1)
  → layer1 Conv 64→128  + Pool   [16×16]
  → ⊕ res1a → res1b  (skip connection)
  → layer2 Conv 128→256 + Pool   [8×8]
  → layer3 Conv 256→512 + Pool   [4×4]
  → ⊕ res2a → res2b  (skip connection)
  → gap    MaxPool 4×4            [1×1]
  → fc     Linear 512→10
  → × 0.125
Output (10 class scores)
```

**~6.6M parameters**

---

## Results

| Epochs | Training Time | Validation Accuracy |
|--------|--------------|-------------------|
| 10     | ~97 seconds (GPU) | **93.66%** |

---

## Key Design Choices

| Feature | Purpose |
|---|---|
| **Split Batch Norm (SplitBN)** | Normalises within 16 sub-batches for stable statistics at any batch size |
| **Residual skip connections** | Lets gradients bypass non-linearities — faster convergence, no vanishing gradients |
| **CELU activation** | Smoother gradient near zero than ReLU, fewer dead neurons |
| **Label smoothing (α=0.2)** | Prevents overconfident predictions, improves generalisation |
| **Two-phase LR schedule** | Linear warmup then decay — reaches a sharp minimum in very few epochs |
| **Nesterov SGD** | Looks ahead before stepping, tracks the loss surface more efficiently |
| **EMA weights** | Exponential moving average of weights used at validation — smoother, better-generalised model |
| **Test-time augmentation** | Averages predictions on original + horizontally flipped inputs at eval time |

---

## Dataset

**CIFAR-10** — 60,000 colour images (32×32) across 10 classes.

The standard train/test split is combined and re-split 80/20:
- **48,000** training images
- **12,000** validation images

Normalised using CIFAR-10 per-channel mean and std (0–255 scale):
- Mean: `[125.31, 122.95, 113.87]`
- Std: `[62.99, 62.09, 66.70]`

---

## File Structure

```
CIFAR10 CNN/
├── FINAL_AdvancedRNN_Architecture.ipynb   # Main notebook (ResNet9)
├── AdvancedRNN_Architecture.ipynb         # Earlier iteration
├── 1_SimpleCIFAR10/
│   ├── train.py                           # Training script (run this)
│   └── model.py                           # Model definition
└── README.md
```

---

## How to Run

### Main Notebook (`FINAL_AdvancedRNN_Architecture.ipynb`)

Open in Jupyter and run cells top to bottom. The dataset loads automatically in **Section 1**:

- **Section 0** — imports (run first)
- **Section 1.1** — defines the preprocessing function
- **Section 1.2** — defines `load_cifar10()` (downloads CIFAR-10 to `./data/` on first run)
- **Section 1.3** — actually runs the download and loads everything into memory; this is the cell that takes a moment

Everything after that (model, training, results) depends on section 1 having run successfully.

### Simple Baseline (`1_SimpleCIFAR10/`)

A standalone script version of the same pipeline. Run it from inside the folder:

```bash
cd 1_SimpleCIFAR10
python train.py
```

CIFAR-10 downloads automatically to `../.data/` on first run. By default it trains for 100 seeds and reports min/max/mean accuracy across runs — if you just want a single run, you can kill it after the first epoch summary prints.

---

## Requirements

- Python 3.11+
- PyTorch
- torchvision
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

---

## Acknowledgements

This project was completed as part of a group assignment. Thanks to my teammates:

- **Jorge Lara Mino**
- **Naratorn Pisedtalasalasai**

---

## References

- He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M. (2019). Bag of tricks for image classification with convolutional neural networks. *CVPR 2019*. https://arxiv.org/abs/1812.01187
- Willaert, J. (2021). How to calculate the mean and standard deviation — Normalizing datasets in PyTorch. *Towards Data Science*.
