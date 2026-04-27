# Pre-Formulated Hypotheses for Ablation Studies

**Project**: Adversarial Robustness in Artistic Style Recognition  
**Team**: Post Team
**Date**: May 3, 2026

---

## Hypothesis 1: Adversarial Training Effectiveness

### Statement
Adversarial training will reduce attack success rates by **at least 50%** compared to standard training when evaluated against PGD attacks with ε=0.03.

### Rationale
The model trained with adversarial examples during training should learn more robust representations. By being exposed to adversarial perturbations during training, the model's loss landscape becomes flatter and less susceptible to small input perturbations. This is based on Madry et al. (2018) findings that adversarial training significantly improves robustness.

### Experimental Design
- **Baseline Model**: Standard training on WikiArt (no adversarial examples)
- **Defended Model**: Adversarial training with α=0.5 mixing ratio
- **Metric**: Attack Success Rate (%) on test set
- **Attack**: PGD with ε=0.03, 20 steps
- **Expected Result**: Defended model ASR < Baseline ASR × 0.5

### What We'll Measure
- Attack success rate at different epsilon values
- Robustness curve (epsilon vs. accuracy)
- Trade-off with clean accuracy (if any)

---

## Hypothesis 2: Style-Dependent Robustness

### Statement
Abstract and impressionist styles will be **at least 20% more robust** to adversarial attacks than representational styles (e.g., realism, photorealism).

### Rationale
Abstract and impressionist styles rely on high-level artistic concepts and global compositional patterns, which are more robust features. Representational styles depend on fine details and local pixel patterns that are more vulnerable to adversarial perturbations. The model learns to extract different features for different styles, and abstract styles' features are inherently less sensitive to small perturbations.

### Experimental Design
- **Test Set**: Grouped by 27 artistic styles in WikiArt
- **Metric**: Attack Success Rate per style (ASR_style)
- **Attack**: FGSM and PGD with ε=0.03
- **Categorization**:
  - Abstract Styles: Abstract, Abstraction, Abstract Expressionism, Cubism, Futurism
  - Impressionist Styles: Impressionism, Post-Impressionism, Pointillism
  - Representational Styles: Photorealism, Realism, Neorealism, Hyperrealism

### What We'll Measure
- Attack success rate per style
- Confidence distribution before/after attacks by style
- Feature visualization for different style categories

### Interpretation
- If hypothesis is true: Abstract styles have lower ASR
- If hypothesis is false: Robustness is uniform across styles OR representational styles are more robust (implies model relies on different features than expected)

---

## Hypothesis 3: Attack Method Efficiency

### Statement
PGD attacks will require **2-3x smaller perturbation magnitudes (epsilon values)** than FGSM to achieve comparable attack success rates due to the iterative optimization process.

### Rationale
FGSM uses a single gradient step, while PGD iteratively optimizes the adversarial perturbation. PGD's iterative nature allows it to find more efficient adversarial examples that fool the model with smaller perturbations. This is the fundamental trade-off between attack complexity and success.

### Experimental Design
- **Attacks**: FGSM vs. PGD (20 steps, step_size=0.001)
- **Metric**: Attack Success Rate (ASR)
- **Epsilon Range**: [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1]
- **Comparison Point**: 50% ASR threshold

### What We'll Measure
- Plot: ASR vs. epsilon for both attacks
- Epsilon required to achieve 50% ASR: ε_PGD vs. ε_FGSM
- Ratio: ε_FGSM / ε_PGD (expected: 2-3)

### Interpretation
- If ratio is 2-3: PGD is significantly more efficient
- If ratio is <2 or >3: Attack efficiency differs from expectations (might indicate model architecture or training issues)

---

## Hypothesis 4 (Bonus): Data Augmentation as Defense

### Statement
Data augmentation-based defense (random crops, rotations, color jitter) will improve robustness by **30-40%** compared to standard training, but **less effectively than adversarial training**.

### Rationale
Data augmentation increases input diversity during training, which can improve robustness. However, it's an indirect defense that doesn't explicitly optimize for adversarial robustness. We expect it to be helpful but inferior to adversarial training.

### Experimental Design
- **Model 1**: Standard training
- **Model 2**: Data augmentation training
- **Model 3**: Adversarial training
- **Metric**: Attack Success Rate at ε=0.03

### Expected Ranking
1. Adversarial Training: Lowest ASR (best)
2. Data Augmentation: Medium ASR
3. Standard Training: Highest ASR (worst)

---

## Pre-Study Questions

Before running experiments, answer these:

1. **Do we have GPU access?** (affects experiment timeline)
2. **Which attack methods can we implement?** (FGSM and PGD are essential; AutoAttack is bonus)
3. **How will we measure human imperceptibility?** (human study, LPIPS distance, other metrics?)
4. **What's our fallback if a hypothesis is rejected?**

---

## Ablation Study Checklist

- [ ] Hypothesis 1: Adversarial training reduces ASR by >50%
- [ ] Hypothesis 2: Abstract styles are 20% more robust
- [ ] Hypothesis 3: PGD needs 2-3x smaller epsilon than FGSM
- [ ] Hypothesis 4: Data augmentation improves robustness 30-40%

---

## Expected Timeline

| Phase | Duration | Hypothesis Testing |
|-------|----------|-------------------|
| Data & Baseline Setup | Week 1-2 | Establish clean accuracy baseline |
| FGSM Implementation | Week 2 | Hypothesis 3 (FGSM component) |
| PGD Implementation | Week 2-3 | Hypothesis 3 (full test) |
| Adversarial Training | Week 3-4 | Hypothesis 1 |
| Defense Comparison | Week 4 | Hypothesis 1 & 4 |
| Style-wise Analysis | Week 4-5 | Hypothesis 2 |
| Visualization & Write-up | Week 5-6 | All hypotheses |

---

## How We'll Handle Unexpected Results

If Hypothesis 2 is rejected (robustness is uniform across styles):
→ Investigation: Perform saliency analysis. Do abstract and representational styles activate different model regions? If yes, features are different but equally robust. If no, the model isn't learning style-specific features.

If Hypothesis 1 shows <50% improvement:
→ Investigation: Increase adversarial training hyperparameters (alpha, epsilon, num_steps). Check if model is actually training on adversarial examples.

If Hypothesis 3 shows PGD is not more efficient:
→ Investigation: Verify PGD implementation. Check gradient computation. Increase PGD iterations.

---

## References

- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR 2018*.
- Goodfellow, I., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. *ICLR 2015*.
- Carlini, C., & Wagner, D. (2016). Towards evaluating the robustness of neural networks. *IEEE S&P 2017*.

---


