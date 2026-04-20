#!/bin/bash
#SBATCH --job-name=adv-style
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/slurm-%j.out
#SBATCH --error=results/logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qiuli_lai@brown.edu

# ── Environment ──────────────────────────────────────────────
module load python/3.11.0
module load cuda/11.8.0

cd ~/adversarial-style-robustness

# Install dependencies (first run only — comment out after)
pip install -r requirements.txt --user -q

# ── Phase 1: Baseline ─────────────────────────────────────────
echo "=== Training Baseline ==="
python experiments/train_baseline.py \
    --data_dir ./data/wikiart \
    --epochs 50 \
    --batch_size 64 \
    --device cuda \
    --output_dir ./results/baseline

# ── Phase 2: Evaluate robustness ──────────────────────────────
echo "=== Evaluating Robustness ==="
python experiments/evaluate_robustness.py \
    --model_path ./results/baseline/best_model.pth \
    --data_dir ./data/wikiart \
    --attack fgsm pgd \
    --output_dir ./results/evaluation \
    --device cuda

echo "=== Done ==="
