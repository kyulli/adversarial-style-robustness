#!/bin/bash
#SBATCH --job-name=adv-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/slurm-eval-%j.out
#SBATCH --error=results/logs/slurm-eval-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qiuli_lai@brown.edu

module load cuda/11.8.0
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=/usr/local/lib64/python3.9/site-packages:$HOME/adversarial-style-robustness:$PYTHONPATH

cd ~/adversarial-style-robustness

# ── Evaluate Adversarial Training Model ───────────────────────
echo "=== Evaluating Adversarial Training Model ==="
python experiments/evaluate_robustness.py \
    --model_path ./results/adversarial_training/best_model.pth \
    --data_dir ./data/wikiart \
    --output_dir ./results/evaluation_adv \
    --device cuda

# ── Evaluate Augmentation Model ───────────────────────────────
echo "=== Evaluating Augmentation Model ==="
python experiments/evaluate_robustness.py \
    --model_path ./results/augmentation/best_model.pth \
    --data_dir ./data/wikiart \
    --output_dir ./results/evaluation_aug \
    --device cuda

echo "=== All evaluations done ==="
