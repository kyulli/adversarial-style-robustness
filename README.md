# Adversarial Robustness in Artistic Style Recognition

**Project for CSCI 1470: Deep Learning (Spring 2026)**

## Overview

This project investigates the robustness of deep learning models in artistic style recognition under minimal input perturbations. While modern convolutional neural networks achieve strong performance on style classification tasks, it is unclear whether these models truly capture high-level artistic concepts or rely on fragile low-level visual cues.

### Research Question
**How vulnerable are CNN-based style classifiers to adversarial perturbations, and what defense mechanisms can improve robustness?**

## Project Goals

1. **Train a baseline CNN** on WikiArt dataset for artistic style classification
2. **Generate adversarial examples** using gradient-based techniques (FGSM, PGD)
3. **Evaluate robustness** by measuring:
   - Minimum perturbation required to flip predictions
   - Robustness variation across artistic styles
   - Human imperceptibility of perturbations
4. **Test defense mechanisms** (adversarial training, data augmentation, etc.)
5. **Analyze and visualize** results to understand model vulnerabilities

## Dataset

**WikiArt Dataset**
- ~80,000 images across 27 artistic styles
- Styles include: Abstract, Baroque, Cubism, Impressionism, Renaissance, etc.
- Download: [WikiArt Official](https://www.wikiart.org/)

## Key Hypotheses (Pre-Formulated)

**Hypothesis 1**: Adversarial training will reduce attack success rates by >50% compared to standard training, because it exposes the model to adversarial examples during training.

**Hypothesis 2**: Abstract/impressionist styles will be more robust than representational styles, because they rely on high-level artistic features rather than low-level pixel patterns.

**Hypothesis 3**: PGD attacks will require 2-3x smaller perturbations than FGSM to succeed due to the iterative nature of PGD.

## Dependencies

See `requirements.txt` for full list. Key libraries:
- PyTorch / TorchVision
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Jupyter Notebook

## Installation

```bash
# Clone the repository
git clone git@github.com:kyulli/adversarial-style-robustness.git
cd adversarial-style-robustness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download WikiArt dataset
python data/download_wikiart.py --output_dir ./data/wikiart
```

## Usage

### Training Baseline Model
```bash
python experiments/train_baseline.py \
    --data_dir ./data/wikiart \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --output_dir ./results/baseline
```

### Generating Adversarial Examples
```bash
python experiments/evaluate_robustness.py \
    --model_path ./results/baseline/best_model.pth \
    --attack fgsm \
    --epsilon 0.03 \
    --data_dir ./data/wikiart
```

### Running Ablation Studies
```bash
python experiments/ablation_studies.py \
    --config ./config.yaml \
    --output_dir ./results/ablations
```

## References

### Key Papers
- Goodfellow et al. (2014): Explaining and Harnessing Adversarial Examples
- Madry et al. (2018): Towards Deep Learning Models Resistant to Adversarial Attacks
- Carlini & Wagner (2016): Towards Evaluating the Robustness of Neural Networks

### Datasets
- WikiArt Dataset: https://www.wikiart.org/

## License

This project is for educational purposes as part of CSCI 1470 at Brown University.

