# Figure Descriptions and Usage Guide

This document provides detailed explanations of each figure for use in the project report and presentation.

---

## Figure 1: Replacement Scenario Comparison

**File:** `fig1_replacement_comparison.png`

**Purpose:** This is the **most important figure** - it shows the main result of the project.

### What It Shows
A bar chart comparing four methods on the **replacement scenario** (training ML models using only synthetic data):

| Method | R² Score | % of Baseline | Color |
|--------|----------|---------------|-------|
| SMOGN | -0.14 | FAILED | Red |
| CTGAN | 0.23 | 36% | Orange |
| Simple Diffusion | 0.17 | 27% | Green |
| **TabDDPM (Ours)** | **0.56** | **87%** | Blue |

### Key Visual Elements
- **Dashed line at R²=0.65**: Shows the baseline (training on real data)
- **Percentage labels inside bars**: Shows what fraction of baseline each method achieves
- **"FAILED" label on SMOGN**: Emphasizes that SMOGN produces negative R² (worse than predicting the mean)

### Key Takeaway
> TabDDPM achieves **87% of baseline performance** using only synthetic data - far better than CTGAN (36%) or simple diffusion (27%). SMOGN fails completely.

### When to Use
- **Main results slide** in presentation
- **Results section** of report
- Whenever comparing method performance

### Suggested Talking Points
1. "Our TabDDPM implementation achieves 87% of baseline using only synthetic data"
2. "This is 2.5x better than CTGAN, the previous best method"
3. "SMOGN fails catastrophically on this complex dataset"

---

## Figure 2: Augmentation Scenario Comparison

**File:** `fig2_augmentation_comparison.png`

**Purpose:** Shows that all methods (except SMOGN) maintain baseline performance when adding synthetic data to original data.

### What It Shows
A bar chart comparing methods on the **augmentation scenario** (original + synthetic data):

| Method | R² Score | Result |
|--------|----------|--------|
| Baseline | 0.65 | Reference |
| SMOGN | -0.14 | FAILED (harmful) |
| CTGAN | 0.63 | 97.8% maintained |
| Simple Diffusion | 0.64 | 98.5% maintained |
| **TabDDPM (Ours)** | **0.64** | **99.1% maintained** |

### Key Takeaway
> For augmentation, all generative methods maintain baseline performance (~98-99%). The difference is in **replacement** where TabDDPM excels.

### When to Use
- Secondary results slide
- To show that synthetic data doesn't hurt when combined with real data
- To contrast with replacement scenario

### Suggested Talking Points
1. "For augmentation, all methods except SMOGN work well"
2. "The real differentiation is in the replacement scenario"
3. "SMOGN actually harms model performance when added to training data"

---

## Figure 3: Privacy-Utility Tradeoff

**File:** `fig3_privacy_utility_tradeoff.png`

**Purpose:** Visualizes the key insight that TabDDPM achieves **both** high utility AND high privacy.

### What It Shows
A scatter plot with:
- **X-axis**: Utility (Replacement R²) - higher is better
- **Y-axis**: Privacy Score (1 - Attack AUC) - higher is better
- **Green zone**: Good privacy region (near 0.5 = random guess)
- **Blue zone**: Good utility region (high R²)

### Method Positions
| Method | Position | Interpretation |
|--------|----------|----------------|
| SMOGN | Low utility, medium privacy | Fails at main task |
| Simple Diffusion | Low utility, high privacy | Private but not useful |
| CTGAN | Medium utility, low privacy | Useful but less private |
| **TabDDPM (Ours)** | **High utility, high privacy** | **Best of both worlds** |

### Key Takeaway
> TabDDPM is the **only method in the top-right quadrant** - achieving both high utility (0.56 R²) and high privacy (AUC = 0.51).

### When to Use
- To explain the dual objectives (utility vs privacy)
- To show TabDDPM's unique position
- Key insight slide

### Suggested Talking Points
1. "Most methods force a tradeoff - either utility or privacy"
2. "TabDDPM achieves both - it's in the ideal top-right quadrant"
3. "The arrow shows our goal: maximize both axes"

---

## Figure 4: Method Comparison Summary

**File:** `fig4_method_summary.png`

**Purpose:** Comprehensive comparison showing all three metrics (replacement, augmentation, privacy) side by side.

### What It Shows
Grouped bar chart with three bars per method:
- **Blue**: Replacement score (% of baseline)
- **Green**: Augmentation score (% of baseline)
- **Purple**: Privacy score

### Key Observations
| Method | Replacement | Augmentation | Privacy | Overall |
|--------|-------------|--------------|---------|---------|
| SMOGN | 0% | 0% | 95% | Poor (fails utility) |
| CTGAN | 36% | 98% | 90% | Medium |
| Simple Diffusion | 26% | 98% | 98% | Medium |
| **TabDDPM** | **87%** | **99%** | **98%** | **Best** |

### Key Takeaway
> TabDDPM has the tallest bars across all metrics - it's the best overall method.

### When to Use
- Overview/summary slide
- When comparing multiple metrics at once
- Executive summary

---

## Figure 5: Training Convergence

**File:** `fig5_training_convergence.png`

**Purpose:** Shows that the model trains stably and converges properly.

### What It Shows
Two panels:
1. **Left**: Total loss over 1000 epochs
2. **Right**: Loss breakdown (Numerical MSE vs Categorical KL)

### Key Observations
- Loss drops rapidly in first 200 epochs
- Converges to stable value around 0.17
- **Numerical loss dominates** (green line) - this is expected since there are 3 numerical columns
- **Categorical loss is small** (red line) - KL divergence for 4 categorical columns

### Key Takeaway
> The model trains stably with no oscillation or divergence. The loss components show the hybrid approach works correctly.

### When to Use
- Methodology section
- To show training stability
- If asked about training process

---

## Figure 6: Diffusion Process Diagram

**File:** `fig6_diffusion_process.png`

**Purpose:** Explains how diffusion models work at a conceptual level.

### What It Shows
Two-row diagram:
1. **Top row (Forward Process)**: Clean data progressively corrupted with noise
   - Blue (clean) -> Light colors -> Red (pure noise)
   - "+noise" labels between steps

2. **Bottom row (Reverse Process)**: Noise progressively removed to generate data
   - Red (noise) -> Light colors -> Blue (clean)
   - "denoise" labels between steps (green arrows)

### Key Elements
- **Color gradient**: Blue = clean data, Red = noise
- **x₀, x₁, x_t, x_{T-1}, x_T**: Timestep notation
- **Bottom legend**: "Numerical: Gaussian noise" and "Categorical: Multinomial diffusion"

### Key Takeaway
> Diffusion models learn to reverse the noise-adding process. For tabular data, we use Gaussian noise for numbers and Multinomial diffusion for categories.

### When to Use
- Introduction/background slide
- Explaining how diffusion works
- Methodology section

### Suggested Talking Points
1. "The forward process gradually adds noise until data becomes pure noise"
2. "The model learns to reverse this - removing noise step by step"
3. "For tabular data, we handle numerical and categorical columns differently"

---

## Figure 7: Architecture Diagram

**File:** `fig7_architecture.png`

**Purpose:** Shows the neural network architecture used in TabDDPM.

### What It Shows
Flow diagram from left to right:

**Inputs (left):**
- Blue box: "Numerical (Gaussian)" - scaled numerical features
- Orange box: "Categorical (Log-prob)" - log-probability representation
- Green box: "Timestep t (Embedding)" - which diffusion step we're at

**Processing (middle):**
- Purple "Concat" box: Combines all inputs
- Stack of cyan boxes: MLP layers (Linear 256 -> ReLU + Dropout -> Linear 256 -> ...)

**Outputs (right):**
- Blue box: "Predicted x₀ (Numerical)" with "MSE Loss"
- Orange box: "Predicted x₀ (Categorical)" with "KL Div Loss"

### Key Takeaway
> The architecture is a simple MLP that takes noisy data + timestep and predicts the original clean data. Different loss functions for numerical (MSE) vs categorical (KL divergence).

### When to Use
- Methodology/approach slide
- Technical details section
- If asked about implementation

---

## Figure 8: Privacy Comparison

**File:** `fig8_privacy_comparison.png`

**Purpose:** Shows that all methods pass the privacy test (membership inference attack).

### What It Shows
Bar chart comparing attack AUC:

| Method | AUC | Interpretation |
|--------|-----|----------------|
| Random Guess | 0.500 | Perfect privacy (theoretical) |
| TabDDPM (Ours) | 0.510 | Essentially random |
| Simple Diffusion | 0.512 | Essentially random |
| SMOGN | 0.525 | Essentially random |
| Privacy Threshold | 0.600 | Above this = concern |

### Key Visual Elements
- **Green shaded zone**: "Safe Zone (Random Guess)" - all bars should be here
- **Hatched bar at 0.6**: Privacy threshold - anything above is concerning
- **"All methods are SAFE"** label

### Key Takeaway
> All methods have AUC close to 0.5 (random guessing), meaning an attacker cannot determine if any specific record was in the training set. **TabDDPM is the most private (lowest AUC).**

### When to Use
- Privacy results slide
- To show synthetic data doesn't leak information
- Addressing privacy concerns

### Suggested Talking Points
1. "A membership inference attack tries to guess if a record was in training data"
2. "AUC of 0.5 means the attacker is just guessing randomly"
3. "All our methods pass this test - no privacy leakage"

---

## Figure 9: Key Results Summary

**File:** `fig9_key_results_summary.png`

**Purpose:** Visual summary card highlighting the three main achievements.

### What It Shows
Three colored boxes with key numbers:

| Box | Color | Number | Meaning |
|-----|-------|--------|---------|
| UTILITY | Green | **87%** | of baseline R² achieved |
| PRIVACY | Blue | **0.51** | attack AUC (= random guess) |
| IMPROVEMENT | Orange | **3.3x** | better than simple diffusion |

### Additional Elements
- Subtitle: "TabDDPM-style diffusion achieves the best of both worlds"
- Bottom box: Lists four key improvements (Log-space ops, KL divergence loss, Gumbel-softmax, Proper posterior)

### Key Takeaway
> Three numbers to remember: **87% utility**, **0.51 privacy**, **3.3x improvement**

### When to Use
- **Title slide or conclusion slide** in presentation
- Executive summary
- Key takeaways

---

## Recommended Figure Usage by Presentation Section

| Section | Recommended Figures |
|---------|---------------------|
| Title/Introduction | fig9 (key results summary) |
| Problem Definition | fig6 (diffusion process) |
| Methodology | fig6, fig7 (process + architecture) |
| Training | fig5 (convergence) |
| Results - Main | fig1 (replacement comparison) |
| Results - Secondary | fig2 (augmentation) |
| Results - Privacy | fig8 (privacy comparison) |
| Key Insight | fig3 (privacy-utility tradeoff) |
| Summary/Conclusion | fig4 (method summary) or fig9 |

---

## Notes for Improvement

### Minor Issues Observed
1. **fig6 & fig7**: Subscript characters (x₀, x₁) may not render perfectly due to font limitations
2. **fig3**: CTGAN privacy score may need verification (currently estimated)

### Potential Enhancements
1. Add confidence intervals to bar charts
2. Create animated version of fig6 for presentation
3. Add feature distribution comparison plots
