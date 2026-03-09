# Precision Effects on Training Dynamics

Research investigation into how reduced numerical precision influences optimization trajectories and training stability in large-scale language models.

---

## Research Motivation

Modern large language models increasingly rely on low-precision formats (e.g., FP8) for efficiency.  
While convergence quality is often evaluated via final loss, less attention is paid to how reduced precision alters the *optimization trajectory itself*.

This project studies how numerical precision affects:

- Gradient magnitude
- Gradient direction
- Parameter scale evolution
- Optimization stability

**One-line takeaway:**  
Reduced precision can preserve loss behavior while still inducing measurable shifts in gradient geometry and optimization trajectory.

---

## Research Questions

1. How do FP8 formats (E4M3 / E5M2) differ from BF16 in training dynamics?
2. Does layer-wise activation quantization shift optimization trajectories?
3. Are gradient-based metrics more sensitive than loss for detecting instability?
4. Do different structural modules (MoE vs Attention) exhibit different precision sensitivity?

---

## Experimental Setup

- Model scale: GPT-style language model and DeepSeek-MoE style architecture
- Baseline precision: BF16
- Low-precision formats: FP8 (E4M3, E5M2)
- Quantization strategies:
  - Multi-layer MoE activation quantization
  - Multi-layer Attention (MHA) activation quantization
  - Single-layer perturbation experiments
- Metrics:
  - Loss
  - Gradient norm
  - Weight norm
  - Gradient direction angle
  - Relative gradient error

---

# Selected Figures and Empirical Findings

---

## 1️⃣ Baseline Training Dynamics (BF16 Reference)

![Baseline Training Dynamics](figures/baseline_training_dynamics.png)

**Observation**

Under BF16 precision:

- Loss decreases smoothly.
- Gradient norm stabilizes after initial decay.
- Weight norm evolves consistently with the learning rate schedule.

**Interpretation**

This establishes a stable optimization reference trajectory.  
All precision perturbation experiments are evaluated relative to this baseline to isolate structural deviations.

---

# 2️⃣ Multi-layer MoE Quantization (FP8)

---

### (a) Optimization Trajectory Comparison

![MoE Comprehensive](figures/moe_comprehensive_comparison.png)

**Key Insight**

- Loss remains close to BF16.
- Gradient norm magnitude remains stable.
- However, gradient deviation accumulates progressively.

This suggests FP8 introduces structural perturbations not immediately visible in scalar loss metrics.

---

### (b) Gradient Direction Shift

![MoE Angle](figures/moe_angle_comparison_between_bf16_fp8.png)

**Key Insight**

The angle between FP8 and BF16 gradients increases over training.

This indicates gradual divergence in optimization trajectory geometry.

MoE layers exhibit moderate sensitivity to reduced precision.

---

### (c) Relative Error Evolution

![MoE Relative Error](figures/moe_relative_errors_between_bf16_fp8.png)

**Key Insight**

- Loss relative error remains small.
- Gradient difference norm grows significantly.

This decoupling suggests that loss alone may underestimate precision-induced instability.

---

# 3️⃣ Multi-layer Attention (MHA) Quantization (FP8)

---

### (a) Optimization Trajectory Comparison

![MHA Comprehensive](figures/mha_comprehensive_comparison.png)

**Key Insight**

- Loss remains stable.
- Gradient deviation grows faster compared to MoE.

Attention layers appear more sensitive to quantization noise.

---

### (b) Gradient Direction Shift

![MHA Angle](figures/mha_angle_comparison_between_bf16_fp8.png)

**Key Insight**

Gradient angle deviation grows more rapidly than in MoE experiments.

This indicates stronger geometric sensitivity in attention modules under reduced precision.

---

### (c) Relative Error Evolution

![MHA Relative Error](figures/mha_relative_errors_between_bf16_fp8.png)

**Key Insight**

Relative gradient error accumulates steadily.

This suggests structural components (e.g., attention mechanisms) may amplify low-precision perturbations more than feed-forward MoE layers.

---

# Cross-Module Comparison

| Component | Loss Stability | Gradient Magnitude | Direction Drift | Sensitivity |
|------------|---------------|-------------------|-----------------|-------------|
| MoE | Stable | Stable | Moderate | Medium |
| Attention | Stable | Stable | Larger | Higher |

---

## Interpretation and Research Implications

Across experiments, a consistent pattern emerges:

- Reduced precision primarily perturbs gradient geometry and optimization trajectory alignment.
- Scalar loss behavior may remain stable while directional drift accumulates.
- Multi-layer perturbations produce cumulative structural effects.
- Attention modules exhibit higher precision sensitivity than MoE layers.

From an ML systems perspective, these findings suggest that:

> Numerical precision affects not only convergence outcomes, but also the geometric structure of optimization trajectories.

Training dynamics analysis may provide earlier and more sensitive instability signals than loss curves alone.

---

## Ongoing Directions

- Deeper analysis of E4M3 vs E5M2 structural differences
- Scaling experiments on larger parameter regimes
- Investigation of precision-aware optimization strategies
- Theoretical connection between rounding bias and trajectory shift

---

## Contact

This repository documents ongoing research exploration in low-precision optimization and efficient ML systems.

Feel free to reach out for academic discussion.
