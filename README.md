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
