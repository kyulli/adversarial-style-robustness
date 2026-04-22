#!/bin/bash
#SBATCH --job-name=adv-defense
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm-defense-%j.out
#SBATCH --error=results/logs/slurm-defense-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qiuli_lai@brown.edu

module load cuda/11.8.0
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/adversarial-style-robustness:$PYTHONPATH
pip install --user numpy --quiet

cd ~/adversarial-style-robustness

# ── Adversarial Training ──────────────────────────────────────
echo "=== Adversarial Training ==="
python experiments/train_defended.py \
    --defense adversarial_training \
    --data_dir ./data/wikiart \
    --epochs 50 \
    --batch_size 64 \
    --device cuda \
    --output_dir ./results/adversarial_training

# ── Augmentation Training ─────────────────────────────────────
echo "=== Augmentation Defense ==="
python experiments/train_defended.py \
    --defense data_augmentation \
    --data_dir ./data/wikiart \
    --epochs 50 \
    --batch_size 64 \
    --device cuda \
    --output_dir ./results/augmentation

# ── Ablation Studies ──────────────────────────────────────────
echo "=== Ablation Studies ==="
python experiments/ablation_studies.py \
    --baseline_path ./results/baseline/best_model.pth \
    --adv_path ./results/adversarial_training/best_model.pth \
    --aug_path ./results/augmentation/best_model.pth \
    --data_dir ./data/wikiart \
    --output_dir ./results/ablations \
    --device cuda

echo "=== All done ==="
