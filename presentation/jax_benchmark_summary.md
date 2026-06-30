# JAX Benchmark Summary

This note summarizes the current PyTorch-vs-JAX probes. The `2du1` benchmark is
an FT-HMC evaluation benchmark for a fixed trained field transform. The `2du2`
benchmark is a smaller U(2) Wilson-action force probe because a full JAX 2du2
FT-HMC backend has not been implemented yet.

## 2du1 FT-HMC Evaluation

- System: `2du1` U(1) FT-HMC evaluation only; no training.
- Checkpoints: existing `base_scaling_train_b3.0_L8_1029` and `base_scaling_train_b3.0_L16_1029`.
- Device: NVIDIA GPU, PyTorch CUDA and JAX CUDA backend.
- Parameters: `beta=train_beta=3.0`, `n_steps=10`, `n_thermalization=10`, `n_configs=64`, seed `1029`.
- Step sizes: `L=8` uses `ft_step_size=0.4`; `L=16` uses `ft_step_size=0.35`.
- PyTorch mode: matches the current `2du1/evaluation/base/compare_fthmc.py --if_compile` policy, compiling only the force-path field transformation and Jacobian callables with `torch.compile(..., backend="inductor")`.
- JAX mode: compiles the full fixed-shape thermalization and run chain with `jax.jit`.

### Timing Results

| L | PyTorch compile time | JAX compile time | PyTorch compiled steady run | JAX compiled steady run | JAX steady speedup |
|---:|---:|---:|---:|---:|---:|
| 8 | 11.70 s | 83.61 s | 24.60 s | 1.42 s | 17.3x |
| 16 | 0.02 s | 82.95 s | 23.84 s | 1.43 s | 16.7x |

The `L=16` PyTorch compile time is a warm-cache measurement from the same Python
process after the `L=8` compiled run. A cold PyTorch compile can be substantially
higher; a standalone `L=8` cold probe measured about 158 s for the first compiled
force call before cache reuse.

### Result Sanity Checks

| L | PyTorch acceptance | JAX acceptance | PyTorch plaquette mean | JAX plaquette mean | PyTorch topo std | JAX topo std |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.750 | 0.828 | 0.7991 | 0.8100 | 0.947 | 0.882 |
| 16 | 0.766 | 0.812 | 0.8127 | 0.8155 | 1.346 | 1.635 |

The chains are short and use different RNG implementations, so exact trajectory
matching is not expected. The acceptance rates and plaquette/topology summaries
are in the same range, which is sufficient for this timing-focused probe.

### 2du1 Conclusion

The JAX backend has a larger one-time compile cost than a warm PyTorch compiled
force path, but its steady-state evaluation runtime is about 17x faster for the
tested `2du1` FT-HMC workloads. For production-style runs with many configurations
or repeated evaluations at the same shape, the compile cost is amortized quickly.

The next useful step is a longer JAX-only production probe, for example
`n_configs=2048`, to measure end-to-end throughput after compile amortization.

## 2du2 U(2) Wilson Force Probe

This is a small migration feasibility probe, not a full 2du2 FT-HMC evaluation.
It compares the standard U(2) Wilson-action force with PyTorch autograd/compile
against a minimal JAX implementation of the same split phase-plus-quaternion group
operations.

- System: `2du2` U(2) standard Wilson action force only.
- Field: random near-identity U(2) field.
- Parameters: `L=8`, `beta=3.0`, `n_repeat=16`.
- PyTorch mode: `torch.compile(..., backend="inductor")` around the repeated force/update loop.
- JAX mode: `jax.jit` around the repeated force/update loop.

### 2du2 Timing Results

| Probe | PyTorch compile+first | JAX compile | PyTorch compiled steady | JAX compiled steady | JAX steady speedup |
|---|---:|---:|---:|---:|---:|
| U(2) Wilson force, L=8 | 7.33 s | 1.35 s | 0.0133 s | 0.00047 s | 28.2x |

### 2du2 Consistency Checks

| Quantity | Max/absolute difference |
|---|---:|
| Wilson action | 1.14e-05 |
| Force | 2.38e-07 |
| Repeated update final field | 2.82e-06 |

### 2du2 Conclusion

The small U(2) probe is encouraging: JAX is faster both at compile time and
steady-state runtime for this isolated Wilson-force kernel. This does not prove a
full 2du2 FT-HMC speedup yet, because the production transform includes the
non-Abelian field transform, manual active-link Jacobian blocks, observables, and
sampler control flow. It does indicate that the core U(2) group/action/force
operations are compatible with JAX and worth porting next if 2du2 evaluation speed
becomes the priority.
