# Ayaan Khan | DS-D | 22I-2066  
## Deep Learning for Perception — Assignment #1

---

## Project overview
Multi-task facial affect recognition using CNN backbones. The task contains:
- **Categorical**: 8-class expression classification  
- **Continuous**: Valence and Arousal regression

Dataset: `C:\Users\Ayaan\Desktop\dl\Dataset` (images + annotations)

---

## Network details, training settings & rationale

### Architectures used (baselines)
- **ResNet-50** (pretrained on ImageNet) — **25.1M** parameters. Good trade-off: robust, widely used, easy to fine-tune.  
- **EfficientNet-B3** (pretrained) — **11.9M** parameters. Modern, efficient, strong accuracy per FLOP.  
- **VGG-16** (pretrained) — **137.4M** parameters. Classic deep CNN; included to show effect of older large models vs modern architectures.  
- **MobileNet-V2** (pretrained) — **3.2M** parameters. Lightweight, mobile friendly baseline.

### Multi-task heads (same head design for all backbones)
- **Shared backbone → two heads**:
  - **Classification head**: `Linear(feature_dim → 512) → BN → ReLU → Dropout(0.5) → Linear(512 → 8)`, CrossEntropyLoss (label_smoothing=0.1).
  - **Regression head**: `Linear(feature_dim → 256) → ReLU → Dropout(0.5) → Linear(256 → 2)`, MSELoss for valence & arousal.

### Training settings used in experiments
- Input size: **224 × 224**  
- Augmentation (on-the-fly): `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, `RandomRotation`  
- Optimizer: **AdamW** (lr tuned per model; common experiments used `lr=3e-4`)  
- Scheduler: **CosineAnnealingLR** (`T_max = epochs`)  
- Mixed precision: **AMP** (`GradScaler` + `autocast`)  
- Batch size: **32** (reduce to 16 if OOM). `num_workers=0` on Windows.  
- Multi-task loss: `loss = CE + λ * MSE` with `λ = 0.5`

### Rationale for baselines
ResNet50 and EfficientNet were primary choices for modern strong baselines; VGG16 and MobileNetV2 were included as an overly large classical model and a very lightweight mobile model to show trade-offs. Transfer learning with ImageNet-pretrained weights was used to accelerate and stabilize training on a ~4k-image dataset.

---

## Comparison of baselines

### Training times (wall-clock)
**50 epochs**
- ResNet-50 — 18 m 40 s  
- EfficientNet-B3 — 20 m 50 s  
- VGG-16 — 30 m 58 s  
- MobileNet-V2 — 18 m 58 s

**25 epochs**
- ResNet-50 — 12 m 22 s  
- EfficientNet-B3 — 12 m 19 s  
- VGG-16 — 17 m 23 s  
- MobileNet-V2 — 11 m 14 s

### Quantitative results (validation set)
**Classification (summary)**

| Model | Num params | acc | f1_macro | kappa | kripp_alpha | auc_roc_ovr | auc_pr_macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 25,087,306 | 0.4950 | 0.4919 | 0.4229 | 0.4224 | 0.8430 | 0.5084 |
| EfficientNet-B3 | 11,882,290 | 0.4725 | 0.4701 | 0.3971 | 0.3962 | 0.8127 | 0.4540 |
| VGG16 | 137,412,682 | 0.3725 | 0.3517 | 0.2829 | 0.2790 | 0.8077 | 0.3827 |
| MobileNet-V2 | 3,213,322 | 0.4800 | 0.4737 | 0.4057 | 0.4037 | 0.8461 | 0.5031 |

**Regression (valence/arousal)**

| Model | val_rmse | aro_rmse | val_corr | aro_corr | val_sagr | aro_sagr | val_ccc |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 0.3714 | 0.3181 | 0.6113 | 0.5349 | 0.7713 | 0.8150 | 0.5445 |
| EfficientNet-B3 | 0.3966 | 0.3367 | 0.5608 | 0.4822 | 0.7550 | 0.8063 | 0.5312 |
| VGG16 | 0.4133 | 0.3430 | 0.4820 | 0.4015 | 0.7250 | 0.7863 | 0.3310 |
| MobileNet-V2 | 0.3859 | 0.3266 | 0.5855 | 0.5260 | 0.7438 | 0.8125 | 0.5553 |

### Brief takeaway
- **ResNet-50** achieves the best overall accuracy & balanced CCC.  
- **EfficientNet-B3** is close in performance with fewer parameters — good accuracy/efficiency trade-off.  
- **MobileNet-V2** is surprisingly competitive for a lightweight model.  
- **VGG-16** overfits on this modest dataset and performs worst.

---

## Transfer learning details
- All backbones initialized with ImageNet weights (`weights="IMAGENET1K_V1"` in PyTorch).  
- Warm-up experiments used head-only training (freeze backbone), then full fine-tuning.  
- Checkpoints saved as `best_<backbone>.pth` (selected by average CCC).

---

## Training graphs
*(Include screenshots of the following plots in the README or final PDF — replace placeholders with actual images saved from the notebook)*

**ResNet-50**
- Loss vs Epoch: `./plots/resnet_loss.png`  
- Validation metrics vs Epoch: `./plots/resnet_val_metrics.png`

**EfficientNet-B3**
- Loss vs Epoch: `./plots/effb3_loss.png`  
- Validation metrics vs Epoch: `./plots/effb3_val_metrics.png`

**VGG-16**
- Loss vs Epoch: `./plots/vgg_loss.png`  
- Validation metrics vs Epoch: `./plots/vgg_val_metrics.png`

**MobileNet-V2**
- Loss vs Epoch: `./plots/mob_loss.png`  
- Validation metrics vs Epoch: `./plots/mob_val_metrics.png`

---

## Continuous metrics: RMSE, CORR, SAGR, CCC — rationale & recommendation

**Definitions**
- **RMSE** — average magnitude of prediction error (lower is better).  
- **CORR** (Pearson) — linear association; indicates trend capture but ignores scale.  
- **SAGR** (Sign Agreement) — fraction of samples where predicted and true signs match (positive/negative); important for coarse affect polarity detection.  
- **CCC** — combines correlation and mean difference; penalizes both lack of correlation and bias.

**Which matters most in the wild?**  
- **CCC** is the preferred single metric because it captures both correlation and bias. **SAGR** is also crucial when polarity (positive/negative) is the main application. RMSE and Pearson are useful complementary statistics.

---

## Qualitative results
Include a small set of correctly and incorrectly classified images per model (saved from the notebook under `./qualitative_results/`). Example placeholders:

- **ResNet-50**  
  - Correct: `./qualitative_results/ResNet50_correct_12.png`, `..._47.png`, `..._103.png`  
  - Incorrect: `./qualitative_results/ResNet50_incorrect_8.png`, `..._29.png`, `..._76.png`

- **EfficientNet-B3** — similar file list  
- **VGG-16** — similar file list  
- **MobileNet-V2** — similar file list

*(Replace with actual images & short notes on failure modes: lighting, occlusion, pose, ambiguous expressions.)*

---

## Short discussion, limitations & future work
- **Why ResNet/EfficientNet did well:** modern architectures generalize better and capture discriminative features.  
- **Why VGG underperformed:** overparameterized; prone to overfitting on small datasets.  
- **Improvements to try:** class rebalancing, Mixup/CutMix, semi/self-supervised pretraining, ensembling, landmark/geometric feature fusion (e.g., dlib/MediaPipe), calibration for probabilities.

---

## Usage & repository structure
**Quick start**
```sh
# clone the repo
git clone https://github.com/AyaanKhan0111/FacialAffectPyTorchCNN
cd FacialAffectPyTorchCNN

# create venv & install requirements (example)
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# run training (example)
python train.py --backbone resnet50 --epochs 25 --batch-size 32
