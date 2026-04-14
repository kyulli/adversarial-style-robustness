# Quick-Start Guide — Adversarial Style Robustness

Everything is set up. Follow these steps in order.

---

## Step 0: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install PyTorch (choose the right CUDA version for your GPU)
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch torchvision

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Step 1: Download WikiArt Dataset

**Option A — HuggingFace (easiest, ~10GB):**
```bash
pip install datasets
python data/download_wikiart.py --output_dir ./data/wikiart --source huggingface
```

**Option B — Kaggle (~13GB):**
```bash
pip install kaggle
# Put your kaggle.json in ~/.kaggle/
python data/download_wikiart.py --output_dir ./data/wikiart --source kaggle
```

**Verify dataset structure:**
```bash
python data/download_wikiart.py --output_dir ./data/wikiart --verify
```

Expected: ~80,000 images in 27 subdirectories (one per style).

---

## Step 2: Train Baseline Model

```bash
python experiments/train_baseline.py \
    --data_dir ./data/wikiart \
    --architecture resnet18 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --output_dir ./results/baseline \
    --device cuda
```

Saves `./results/baseline/best_model.pth` and `training_history.json`.

---

## Step 3: Evaluate Robustness (Baseline)

```bash
python experiments/evaluate_robustness.py \
    --model_path ./results/baseline/best_model.pth \
    --data_dir ./data/wikiart \
    --attack fgsm pgd \
    --epsilon 0.01 0.02 0.03 0.05 0.1 \
    --output_dir ./results/evaluation
```

Outputs: per-epsilon CSVs with accuracy, attack success rate, L2/Linf, PSNR.

---

## Step 4: Train Defended Models

**Adversarial Training (Defense 1):**
```bash
python experiments/train_defended.py \
    --defense adversarial_training \
    --data_dir ./data/wikiart \
    --epochs 50 \
    --attack_method pgd \
    --epsilon 0.03 \
    --alpha 0.5 \
    --output_dir ./results/adversarial_training \
    --device cuda
```

**Data Augmentation (Defense 2):**
```bash
python experiments/train_defended.py \
    --defense data_augmentation \
    --data_dir ./data/wikiart \
    --epochs 50 \
    --output_dir ./results/augmentation \
    --device cuda
```

---

## Step 5: Ablation Studies

Compare all three models against each other:
```bash
python experiments/ablation_studies.py \
    --baseline_path ./results/baseline/best_model.pth \
    --adv_path ./results/adversarial_training/best_model.pth \
    --aug_path ./results/augmentation/best_model.pth \
    --data_dir ./data/wikiart \
    --output_dir ./results/ablations
```

Produces robustness curve plots and per-style heatmaps.

---

## Step 6: Visualizations

```bash
python experiments/generate_visualizations.py \
    --model_path ./results/baseline/best_model.pth \
    --data_dir ./data/wikiart \
    --output_dir ./results/visualizations \
    --n_show 4
```

Generates: clean vs adversarial image pairs, saliency maps.

---

## File Map

```
adversarial-style-robustness/
├── data_loader.py              # WikiArtDataset + get_dataloaders()
├── attacks.py                  # Combined FGSM + PGD (standalone)
├── attacks/                    # Modular attack package
│   ├── fgsm.py
│   ├── pgd.py
│   └── utils.py                # ASR, perturbation metrics, PSNR
├── defenses/
│   ├── adversarial_training.py # AdversarialTrainer
│   ├── data_augmentation.py    # AugmentationTrainer
│   └── preprocessing.py        # Gaussian blur / quantization
├── models/
│   ├── baseline_model.py       # StyleClassifier (ResNet-18/50)
│   ├── defended_model.py       # Checkpoint loader + preprocessing wrapper
│   └── utils.py                # count_parameters, evaluate_accuracy
├── experiments/
│   ├── train_baseline.py       # Phase 1
│   ├── evaluate_robustness.py  # Phase 2
│   ├── train_defended.py       # Phase 3
│   ├── ablation_studies.py     # Phase 4
│   └── generate_visualizations.py
├── data/
│   ├── download_wikiart.py     # HuggingFace or Kaggle download
│   └── preprocess.py           # Validation, stats, class distribution
└── results/                    # Auto-created subdirs for all outputs
```

---

## Hypothesis Checklist

| # | Hypothesis | How to test |
|---|-----------|-------------|
| 1 | Adversarial training reduces ASR by >50% | Compare `evaluation/` vs adversarial training results |
| 2 | Abstract/impressionist styles more robust | Per-style ASR in `ablations/` heatmaps |
| 3 | PGD needs 2-3× smaller ε than FGSM to succeed | Compare FGSM vs PGD robustness curves |

---

## Tips

- Start with `--max_batches 20` flag for fast debug runs before full evaluation.
- If you don't have a GPU, set `--device cpu` (training will be slow — consider fewer epochs or a smaller dataset subset).
- TensorBoard logs go to `./results/logs/` — run `tensorboard --logdir ./results/logs` to monitor training.
