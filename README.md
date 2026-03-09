# Precision Effects on Training Dynamics

An empirical investigation into how reduced numerical precision influences optimization trajectories and stability signals in LLM training.

---

## Author

Zhao Chenran  
The Chinese University of Hong Kong, Shenzhen

---

## Overview
Low-precision training (e.g., FP8) is increasingly adopted for efficiency in large-scale models.  

While convergence is typically evaluated using final loss, less attention is paid to how reduced precision alters the *optimization trajectory itself*.

This repository documents an ongoing empirical study of how reduced numerical precision affects training dynamics in DeepSeek-MoE models.

Rather than focusing on final benchmark performance, this project examines how reduced precision changes:

- gradient magnitude,
- gradient direction,
- trajectory-level deviation from BF16 reference behavior,
- and the visibility of early instability signals.

The central motivation is that loss alone may not be sufficient to characterize numerical stability during low-precision training.

---

## Research Questions

This project is organized around the following questions:

1. How do FP8 formats (E4M3 / E5M2) differ from BF16 in optimization dynamics?
2. Does multi-layer activation quantization introduce cumulative trajectory shift?
3. Are gradient-based metrics more sensitive than loss in detecting precision-induced deviation?
4. Do different structural components (MoE vs Attention) respond differently to reduced precision?

---

## Experimental Scope

All experiments are conducted on a DeepSeek-MoE architecture.

### Numerical formats
- BF16 (reference baseline)
- FP8 E4M3
- FP8 E5M2

### Quantization settings
- Multi-layer MoE activation quantization
- Multi-layer Attention (MHA) activation quantization

### Metrics analyzed
- Loss
- Gradient norm
- Gradient direction angle
- Relative gradient difference norm

### Scope note
Weight norm is included only in the BF16 baseline panel as a reference for baseline training dynamics.  
This repository does **not** claim comparative parameter-scale analysis under quantized settings.

---

## Why This Study Matters

Reduced precision is often evaluated through final convergence quality or efficiency gains.  
However, two training runs can exhibit similar loss behavior while following meaningfully different optimization trajectories.

This project focuses on that gap:  
whether reduced precision perturbs training geometry before it visibly changes scalar loss.

---

# Selected Figures and Empirical Findings

---

## 1️⃣ Baseline Training Dynamics (BF16 Reference)

![Baseline Training Dynamics](figures/baseline_training_dynamics.png)

**Observation**

Under BF16 precision:

- loss decreases smoothly,
- gradient norm stabilizes after early decay,
- weight norm evolves consistently with the learning rate schedule.

**Interpretation**

This panel defines the reference training trajectory used throughout the repository.  
Subsequent reduced-precision experiments are interpreted relative to this baseline behavior rather than in isolation.

---

## 2️⃣ Multi-layer MoE Quantization (FP8)

### (a) Optimization Trajectory Comparison

![MoE Comprehensive](figures/moe_comprehensive_comparison.png)

**Observation**

- Loss remains close to BF16.
- Gradient norm magnitude remains relatively stable.
- Gradient deviation accumulates progressively across iterations.

**Interpretation**

This suggests that reduced precision can alter optimization behavior even when scalar loss remains well aligned with baseline.

At minimum, the results indicate that loss alone does not fully characterize the effect of low precision on training dynamics.

---

### (b) Gradient Direction Shift

![MoE Angle](figures/moe_angle_comparison_between_bf16_fp8.png)

**Observation**

The angle between FP8 and BF16 gradients increases over training.

**Interpretation**

This indicates gradual directional drift in optimization updates under reduced precision.

In the MoE setting, the perturbation appears cumulative but still compatible with stable loss behavior over the observed training window.

---

### (c) Relative Error Evolution

![MoE Relative Error](figures/moe_relative_errors_between_bf16_fp8.png)

**Observation**

- Loss relative error remains small.
- Relative gradient deviation becomes substantially larger.

**Interpretation**

This decoupling suggests that gradient-based metrics provide more sensitive evidence of precision-induced deviation than loss curves alone.

---

## 3️⃣ Multi-layer Attention (MHA) Quantization (FP8)

### (a) Optimization Trajectory Comparison

![MHA Comprehensive](figures/mha_comprehensive_comparison.png)

**Observation**

- Loss remains stable.
- Gradient deviation grows faster than in the MoE setting.

**Interpretation**

Compared with MoE quantization, attention-layer quantization appears to introduce stronger perturbations into training dynamics.

This does not imply immediate divergence, but it does suggest higher sensitivity of attention modules to reduced precision.

---

### (b) Gradient Direction Shift

![MHA Angle](figures/mha_angle_comparison_between_bf16_fp8.png)

**Observation**

Gradient angle deviation increases steadily and reaches larger values than in MoE experiments.

**Interpretation**

This indicates stronger geometric sensitivity in attention modules under FP8 activation quantization.

A plausible interpretation is that structural differences across modules affect how quantization noise accumulates during training.

---

### (c) Relative Error Evolution

![MHA Relative Error](figures/mha_relative_errors_between_bf16_fp8.png)

**Observation**

Relative gradient error accumulates consistently across training.

**Interpretation**

This reinforces the broader pattern that reduced precision perturbs gradient-level behavior more clearly than loss-level behavior.

---

# Cross-Module Comparison

| Component | Loss Stability | Gradient Magnitude | Direction Drift | Sensitivity |
|------------|---------------|-------------------|-----------------|-------------|
| MoE | Stable | Stable | Moderate | Medium |
| Attention | Stable | Stable | Larger | Higher |

---

## What Can Be Safely Concluded

Based on the current experiments on DeepSeek-MoE:

- Reduced precision perturbs gradient geometry more clearly than loss behavior.
- Multi-layer activation quantization introduces cumulative deviation in optimization trajectory.
- Attention modules appear more sensitive than MoE layers under the tested settings.
- Gradient-based measurements provide more informative stability signals than loss alone.

These conclusions are empirical and limited to the current experimental setup.

---

## What Remains Open

This repository does **not** claim a complete mechanistic explanation of FP8 instability.

Open questions include:

- Why can directional drift accumulate while loss remains aligned?
- What is the relationship between gradient deviation and long-term convergence behavior?
- How do E4M3 and E5M2 differ structurally beyond scalar error magnitude?
- How should optimization diagnostics be adapted for reduced-precision training?

---

## Ongoing Directions

- Deeper comparison of E4M3 vs E5M2
- More systematic trajectory-level diagnostics
- Precision-aware optimization adjustments
- Larger-scale follow-up experiments

---

## Repository Note

This repository is intended as a research presentation artifact rather than a full code release.

It summarizes experimental structure, selected figures, and empirical interpretations from an ongoing project on reduced-precision training dynamics.
