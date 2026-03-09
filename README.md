# Precision Effects on Training Dynamics

## Research Theme

This repository documents my ongoing research on how numerical precision and layer-wise activation quantization affect optimization trajectories and training stability in large-scale language models.

Rather than focusing on benchmark performance, this study investigates how reduced precision alters gradient behavior, parameter evolution, and convergence dynamics.

---

## Research Questions

- How does FP8 (E4M3 / E5M2) differ from BF16 in training dynamics?
- Does layer-wise activation quantization shift optimization trajectories?
- Are gradient-related metrics more sensitive than loss for detecting instability?
- How does multi-layer quantization compare with single-layer perturbation?

---

## Experimental Axes

**Precision Format**
- BF16 (baseline)
- FP8 E4M3
- FP8 E5M2

**Quantization Location**
- MoE layers
- MHA layers

**Quantization Depth**
- Single-layer activation quantization
- Multi-layer activation quantization (0–8 layers)

**Metrics Tracked**
- Training loss
- Gradient difference norm
- Relative error
- Gradient direction alignment
- Weight norm
- Learning rate schedule

---

## Key Observations

### 1. Multi-layer Quantization Shifts Optimization Trajectory
Even when loss remains close to baseline, gradient-related metrics show increasing deviation, suggesting trajectory-level changes in optimization.

### 2. Single-layer Quantization Preserves Alignment
Single-layer activation quantization introduces minimal directional deviation, indicating limited impact on global training dynamics.

### 3. E4M3 Aligns More Closely with BF16 than E5M2
E4M3 consistently exhibits smaller gradient deviation and relative error compared to E5M2.

### 4. Loss Alone Is Insufficient as an Early Instability Indicator
Gradient difference norm and directional deviation often increase before noticeable divergence appears in the loss curve.

### 5. Weight Norm Closely Follows Learning Rate Scheduling
Parameter scale evolution strongly correlates with warmup and decay phases, reinforcing the coupling between effective step size and optimization dynamics.

---

## Selected Figures


### Baseline Training Dynamics (BF16 Reference)

![Baseline Training Dynamics](figures/baseline_training_dynamics.png)

**Observation:** Under BF16, loss, gradient norm, and weight norm evolve smoothly and consistently with the learning rate schedule.

**Interpretation:** This panel establishes a reference optimization trajectory. Subsequent precision perturbations will be compared against this baseline to identify deviations in dynamics beyond nominal behavior.

### Multi-layer MoE Quantization
![MoE Comparison](figures/moe_comprehensive_comparison.png)
![MoE Comparison](figures/moe_angle_comparison_between_bf16_fp8.png)
![MoE Comparison](figures/moe_relative_errors_between_bf16_fp8.png)

### Multi-layer MHA Quantization
![MHA Comparison](figures/mha_comprehensive_comparison.png)
![MHA Comparison](figures/mha_angle_comparison_between_bf16_fp8.png)
![MHA Comparison](figures/mha_relative_errors_between_bf16_fp8.png)



---

## Interpretation and Ongoing Hypotheses

My current interpretation is that reduced precision modifies optimization trajectories before it significantly alters loss convergence.

In particular:
- Gradient-related deviations may serve as earlier instability signals than loss.
- Multi-layer activation quantization accumulates directional bias.
- Numerical format influences effective update behavior even under identical learning rate schedules.

These hypotheses are under ongoing investigation.

---

## Current Limitations

- Spectral analysis of gradient structure is not yet included.
- Full theoretical explanation of trajectory shifts remains open.
- This repository excludes internal infrastructure and unreleased project components.

---

## Future Directions

- Investigate whether gradient alignment metrics can predict divergence.
- Analyze interaction between MoE sparsity and precision.
- Study precision-induced changes in effective learning rate.
- Extend experiments to additional model scales.

---

## Author

Chenran Zhao  
Undergraduate researcher focused on optimization and training dynamics in large-scale ML systems.
