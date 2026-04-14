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

## Project Structure

```
adversarial-style-robustness/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Configuration for experiments
│
├── data/
│   ├── download_wikiart.py           # Script to download WikiArt dataset
│   ├── preprocess.py                 # Data preprocessing and splitting
│   └── dataloader.py                 # PyTorch DataLoader setup
│
├── models/
│   ├── baseline.py                   # Baseline CNN architecture (ResNet-18)
│   ├── defended_model.py             # Model with adversarial training
│   └── utils.py                      # Model utilities
│
├── attacks/
│   ├── fgsm.py                       # Fast Gradient Sign Method
│   ├── pgd.py                        # Projected Gradient Descent
│   ├── autoattack.py                 # AutoAttack wrapper (optional)
│   └── utils.py                      # Attack utilities
│
├── defenses/
│   ├── adversarial_training.py       # Adversarial training defense
│   ├── data_augmentation.py          # Augmentation-based defense
│   └── preprocessing.py              # Input preprocessing defense
│
├── experiments/
│   ├── train_baseline.py             # Train baseline model
│   ├── train_defended.py             # Train defended models
│   ├── evaluate_robustness.py        # Robustness evaluation
│   ├── ablation_studies.py           # Ablation study experiments
│   └── generate_visualizations.py    # Create result visualizations
│
├── results/
│   ├── metrics.csv                   # Quantitative results
│   ├── logs/                         # Training logs
│   └── visualizations/
│       ├── adversarial_examples/     # Adversarial perturbations
│       ├── saliency_maps/            # Model attention maps
│       └── robustness_curves.png     # Robustness vs. perturbation
│
├── notebooks/
│   ├── eda.ipynb                     # Exploratory data analysis
│   └── results_analysis.ipynb        # Results visualization
│
└── docs/
    ├── hypothesis.md                 # Pre-formulated hypotheses
    └── checkpoint_notes.md           # Notes from TA meetings
```

## Key Hypotheses (Pre-Formulated)

**Hypothesis 1**: Adversarial training will reduce attack success rates by >50% compared to standard training, because it exposes the model to adversarial examples during training.

**Hypothesis 2**: Abstract/impressionist styles will be more robust than representational styles, because they rely on high-level artistic features rather than low-level pixel patterns.

**Hypothesis 3**: PGD attacks will require 2-3x smaller perturbations than FGSM to succeed due to the iterative nature of PGD.

## Experimental Plan

### Phase 1: Baseline Setup (Week 1-2)
- [ ] Download and preprocess WikiArt dataset
- [ ] Implement data loading pipeline
- [ ] Train baseline ResNet-18 on style classification
- [ ] Evaluate baseline accuracy

### Phase 2: Adversarial Attacks (Week 2-3)
- [ ] Implement FGSM attack
- [ ] Implement PGD attack
- [ ] Generate adversarial examples
- [ ] Measure attack success rates and perturbation magnitudes

### Phase 3: Defense Mechanisms (Week 3-4)
- [ ] Implement adversarial training
- [ ] Implement data augmentation-based defense
- [ ] Train defended models
- [ ] Compare robustness across defenses

### Phase 4: Ablation Studies & Analysis (Week 4-5)
- [ ] Style-wise robustness analysis
- [ ] Perturbation type comparison
- [ ] Defense mechanism ablation
- [ ] Visualization and interpretation

### Phase 5: Finalization (Week 5-6)
- [ ] Generate poster
- [ ] Write final report
- [ ] Prepare presentation materials

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
git clone https://github.com/YOUR_USERNAME/adversarial-style-robustness.git
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

## Results

### Baseline Performance
- Top-1 Accuracy: [To be filled]
- Top-5 Accuracy: [To be filled]

### Robustness Results
See `results/metrics.csv` and visualizations in `results/visualizations/`

## Team Members
- Deepak Kulkarni 
- Qiuli Lai

## References

### Key Papers
- Goodfellow et al. (2014): Explaining and Harnessing Adversarial Examples
- Madry et al. (2018): Towards Deep Learning Models Resistant to Adversarial Attacks
- Carlini & Wagner (2016): Towards Evaluating the Robustness of Neural Networks

### Datasets
- WikiArt Dataset: https://www.wikiart.org/

## License

This project is for educational purposes as part of CSCI 1470 at Brown University.

## Contact

For questions or discussions about this project, reach out to the team or contact your TA mentor.

---

**Last Updated**: [Date]
**Status**: In Progress
