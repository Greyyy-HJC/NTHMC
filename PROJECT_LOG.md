# PROJECT_LOG.md

This is an append-only development history for NTHMC.

## 2026-07-13

- Removed the separate `src/nthmc/training` package and made PyTorch the sole training implementation under `nthmc.core`, `nthmc.u1`, and `nthmc.u2`; the JAX field transforms are now runtime-only for evaluation and HMC, and Optax is no longer a project dependency.
- Made Torch `if_check_jac=True` keep checking manual vs autograd Jacobian on every transformed-force step for U(1)/U(2), instead of a one-shot startup probe that then disabled the flag; leave `if_check_jac=False` for normal fast training.
- Switched the production U(1)/U(2) `model_training` entrypoints back to a PyTorch eager backend under `nthmc.core`, `nthmc.u1`, and `nthmc.u2`, while retaining JAX for gauge generation, evaluation, and HMC; training now writes resumable `.pt` plus canonical JAX-readable `.npz` checkpoints.
- Added fixed masked global batches, Fabric DDP numerator/count gradient scaling, one shared AdamW per transform, and differentiable per-sample inverse early stopping with mean/max iteration diagnostics; identity transforms stop after one update.
- Validated U(1)/U(2) GPU training and a two-process CPU DDP smoke. The exact U(2) L4/beta-1/batch-16/8-subset epoch took 127.96 seconds internally and 136.64 seconds wall at 2.65 GB host RSS, essentially matching the historical PyTorch baseline while avoiding JAX compilation failure.
- The matching U(1) L16/beta-1/batch-64/8-subset epoch took 5.63 seconds internally and 10.66 seconds wall at 2.20 GB host RSS; inverse diagnostics reported two iterations per subset after the first epoch.
- Rejected local `torch.compile(dynamic=True)` because higher-order autograd hit donated-buffer/create-graph incompatibility. Rejected the TensorFlow 2.21 non-XLA spike after its first trace failed to finish in 422.61 seconds and crossed the 8 GB host-memory gate (8.49 GB RSS, about 10.24 GiB GPU); the isolated TensorFlow environment and prototype were removed.
- Restored the U(2) base CNN to the pre-JAX 18-channel compact scalar loop input (6 plaquette plus 12 rectangle channels), matching the 57,888-parameter PyTorch baseline; incompatible 24-channel JAX checkpoints now fail with an explicit shape error.
- Benchmarked the current L4/beta-1/batch-16 U(2) training workload against the last PyTorch commit (`e3847a9`) on one RTX 3060: PyTorch completed one epoch in 142.18 seconds, while the JAX cold compile was killed after 573.52 seconds at 52.85 GB peak host RSS before producing a cache entry.
- Reduced U(1)/U(2) JAX training compilation graphs by scanning stacked subset parameters, keeping train/eval batches at fixed masked shapes, reusing dynamic-beta/tolerance JIT callables, prewarming timed training graphs, and enabling per-system persistent compilation caches.
- Split inverse execution into a reverse-mode-safe bounded `scan` with conditional early-idling for training and a true dynamically terminating `while_loop` for public/diagnostic calls; added per-sample convergence and iteration-count diagnostics while keeping checkpoint pytrees unchanged.
- Rewrote U(1)/U(2) JAX `models.py` into a readable LocalNet-style layout (`LocalNet.init`/`apply`, `choose_model(tag) -> model class`) and dropped all non-`base` model tags; `FieldTransformation` selects the CNN via `self.model = choose_model(model_tag)`.
- Kept unit tests lightweight on CPU: L=2, `n_subsets=1`, no FT-HMC/HMC chain smokes, no JIT jac compile, no data-parallel training smokes.
- Added an isolated U(2) manual-logdet regression test that compares an eight-subset nontrivial transform against the full field Jacobian in float64, plus a float32 near-identity absolute-tolerance check.

## 2026-06-30

- Migrated the active codebase to JAX-only: U(1)/U(2) observables, HMC, FT-HMC entrypoints, training scripts, shell workflows, dependencies, and tests no longer use the previous training/evaluation stack.
- Switched model checkpoints to JAX `.npz`, removed incompatible old model artifacts, and made `2du1`/`2du2` training use Optax with single-device JAX execution.
- Restored the nontrivial U(2) neural FT path in JAX with attached plaquette/rectangle loops and analytic tangent-propagated active-link Jacobian blocks, avoiding local autodiff Jacobian construction.
- Kept autodiff Jacobian construction behind `if_check_jac=True` for U(1)/U(2) diagnostic checks only; normal training and evaluation use the analytic Jacobian paths.
- Removed the U(1) `if_identity_init` knob and switched U(1) models to the same default identity-start output gate pattern as U(2); added tolerance-checked inverse iteration with diagnostics and tunable `inverse_max_iters`/`inverse_tol` for U(1)/U(2) training.
- Added opt-in local-device JAX `pmap` training via `--data_parallel` for U(1)/U(2), removed leftover no-op JAX compile wrappers, and updated U(2) PBS generation scripts away from Torch/Fabric-era launch flags.
- Removed stale JAX migration compatibility shims and no-op training flags, updated local PBS training scripts for single-node JAX GPU runs, and cleaned current presentation docs that still referenced old Torch/Fabric-era CLI behavior.
- Kept the historical benchmark summary in `presentation/jax_benchmark_summary.md`; updated `README.md` and `SPEC.md` to describe the current JAX-only structure.

## 2026-06-30 Historical JAX Probe

- Added an experimental U(1) JAX FT-HMC backend in `src/nthmc/u1/jax_backend.py`, including JAX observables/action/force, frozen PyTorch checkpoint conversion, JAX CNN inference for `base`/`addcos`, analytic Jacobian logdet, and a JIT-compiled thermalization/run chain.
- Added `2du1/evaluation/jax/compare_fthmc.py` with benchmark JSON output under the new `2du1/evaluation/jax` workspace, CUDA-wheel library-path bootstrapping before JAX import, and matching benchmark JSON output in the existing PyTorch `2du1/evaluation/base/compare_fthmc.py` baseline.
- Added focused U(1) JAX backend tests and documented the JAX path in `SPEC.md`; deferred 2du2 JAX migration until 2du1 benchmarks show a clear steady-state speedup.
- Added `presentation/jax_optimization.ipynb`, a checkpoint-free PyTorch-vs-JAX notebook that demonstrates U(1) forward/Jacobian/force consistency and the GPU JIT steady-state speedup on the transformed-force path.
- Added `presentation/jax_benchmark_summary.md` summarizing L=8/L=16 PyTorch-compiled versus JAX FT-HMC evaluation benchmarks, plus a small 2du2 U(2) Wilson-force PyTorch-vs-JAX probe with compile-time caveats and steady-state speedups.

## 2026-06-17

- Think of translate to tensorflow with jit.
- Think of qex (in nim language), the problem is on neural network, we don't have torch there, but we have adam in qex.
- For the alignment term, add square on it to encourage both align and anti-align.

## 2026-06-17

- Fixed L=16 search probe submission hygiene in `2du2/evaluation/l16_search`: probe jobs now rely on expanded `qsub -o` log paths and pass a per-step `--output_suffix` so concurrent `FT_STEP_SIZE` probes for the same checkpoint/beta do not overwrite each other's topology and acceptance dumps.
- Added queue-submission guards so L=16 T1 eval jobs require an explicitly probed `FT_STEP_SIZE_10`, and `submit_pending_eval.sh` no longer submits T1 or baseline metrics by default before the relevant probe/rerun prerequisites are complete.
- Added `2du2/model_training/sub_gen_d2.sh` to generate the L=16 beta=8.0 seed-1029 T1 training screen for the existing `cap` and `mscap` D2 architecture candidates.

## 2026-06-14

- Added `presentation/optimization_plan.md`, an L=16 U(2) FTHMC architecture-search runbook covering correctness gates, tiered compute budgets, `add_folder.sh` eval workspaces, mandatory `FT_STEP_SIZE` probes, and R_gamma/R_deltaQ win criteria for iterations on `models.py` and `field_transform.py`.
- Added a new "Optimization directions (candidate families)" section to `presentation/optimization_plan.md` defining four orthogonal search directions (D1 receptive-field/multiscale, D2 coefficient caps + gate gain, D3 force-tail loss shaping via `--loss_weights`, D4 topology-directed loss), rewrote the suggested search order to reference D1-D4 cheapest-first, renumbered later sections, fixed section cross-references, and de-duplicated the notebook-config block.

## 2026-06-08

- Added `presentation/model.md` summarizing U(2) model feature ablations at beta=10.0, L=32, seed 1029; kept the current `src/nthmc/u2/models.py` surface to `base`, `wide`, `cap`, and `mscap` after `mscap` outperformed the earlier wide/split/cap combination.
- Added `presentation/alignment.md` summarizing the U(2) topology-alignment diagnostics and short alignment-loss ablations, documenting why the alignment term remains diagnostic-only instead of part of the training loss.

## 2026-06-09

- Added `2du2/evaluation/replot_autocorr.py` to recompute and redraw U(2) autocorrelation overlays for matched `hmc`/`base` topo dump pairs without rerunning evaluation jobs, outputting PDFs under `2du2/evaluation/plots_recomputed`.
- Documented the no-rerun redraw command in `README.md` for quick `(L, beta, nsteps, seed)` filtered comparisons.

## 2026-05-30

- Replaced the U(2) `identity_init` parameter-crippling scheme with a ReZero-style `_LayerScale` output gate. Every U(2) model (`base`, `wide`, `residual`, `dilated`, `mlp`, `flexcap`) now keeps healthy default convolution initialization and starts as the exact identity transform via a zero-initialized per-channel output gate that ramps up during training; the gate's fixed `gain` is the per-variant ramp knob. This fixes the deeper variants' early tanh saturation (dead/stalled or overshooting training) caused by `identity_init` setting every parameter (including the LayerScale) to `N(0, 0.001)`.
- Removed the now-unused `identity_init`/`init_std` knobs from `src/nthmc/u2/field_transform.py`, `2du2/model_training/train.py`, `2du2/evaluation/base/compare_fthmc.py`, and the `2du2` submission scripts (`sub_gen.sh`, `run.sh`, `sub_debug.sh`, `scripts/run_scaling.sh`); updated `tests/test_u2.py` to perturb models explicitly for property-based tests. The `base` model gains a no-op-at-init output gate (48 scalars) but is otherwise unchanged.
- TODO: mirror the same `_LayerScale`/identity-start refactor into the parallel U(1) stack (`src/nthmc/u1/`, `2du1/`) to restore u1/u2 symmetry; deferred this pass to keep the change scoped to U(2).
- Disabled the U(2) topology-alignment contribution in `loss_fn` directly (kept the old alignment-loss path commented for quick rollback), so training/evaluation no longer pays the extra per-batch topology-gradient autograd cost from the loss path.
- Kept per-epoch U(2) diagnostics unchanged, including `force_topo_cos` computation/printing, so alignment observability remains available while the training objective stays force-only.
- Refactored `src/nthmc/u2/field_transform.py` for readability without changing behavior by routing transformed-force helpers through a single core path and moving bulky training-diagnostics collection/formatting into `src/nthmc/u2/field_transform_diagnostics.py`.

## 2026-05-28

- Kept the U(2) autocorrelation definition with fixed-volume normalization, `A(δ)=1-<ΔQ^2(δ)>/(2V)`, and updated the `presentation/2du2_scaling.ipynb` gamma ratio convention to report gain as `gamma_HMC / gamma_FT-HMC` (so FT-HMC improvement appears as values greater than 1).

## 2026-05-27

- Reworked U(2) observable autocorrelation estimates to follow the U(1)-style form with smaller statistical uncertainties, and started evaluation jobs to compare the revised uncertainty behavior.
- Switched the U(2) autocorrelation normalization from sample variance to fixed `2*volume` (`A(δ)=1-<ΔQ^2(δ)>/(2V)`) and aligned the 2du2 scaling notebook gamma helpers with this definition to reduce `R_gamma` noise while keeping the 2du1-style workflow.
- Added a U(2) field-transform training loss term that aligns the transformed force with the topological-charge gradient, with training jobs running to check whether the alignment term lowers the loss.
- Clarified and refined the topology-alignment idea in U(2) training: treat it as a training-time bias (not a physics constraint) to encourage transformed-force updates along topology-changing directions, switch the topology-gradient path to a differentiable soft proxy `Q_soft = sum(sin(theta) + 0.3*sin(2*theta)) / (2*pi)` to avoid near-zero gradients from wrapped-angle cancellation, and use `-mean(cos^2)` so the alignment term rewards nonzero force projection on `grad Q_soft`; the expected benefit is reduced topological freezing (lower critical slowing down in `Q`) rather than guaranteed improvement in total efficiency.

## 2026-05-12

- Set `dynamic=False` in `src/nthmc/u2/field_transform.py` compile options because `dynamic=True` repeatedly triggered `torch/utils/_sympy/interp.py` `pow_by_natural` warnings and substantially increased compile time during U(2) runs.

## 2026-05-09

- Added a browser-friendly `presentation/Field_transform.html` version of the field-transform derivation and linked it from the README.
- Renamed 2du2 `model_training/plots` training-loss PDFs from the `cnn_loss_train_*` prefix to `debug_loss_train_*` for consistency with debug submission artifacts.

## 2026-05-07

- Added an opt-in U(2) training `--checkpoint_delta` mode that activation-checkpoints `compute_delta` to reduce peak GPU memory while preserving the exact training loss and gradients.

## 2026-05-06

- Reduced U(2) validation/diagnostic GPU memory use by disabling higher-order transformed-force graphs outside training steps and clearing optimizer gradients with `set_to_none=True`.
- Migrated the U(2) gauge convention to `U^G_{x,mu} = G_x U_{x,mu} G^dagger_{x+mu}`, updated non-Abelian plaquette/rectangle loop ordering and tangent propagation, and added covariance tests.
- Updated the U(2) field-transform CNN input to use six gauge-invariant scalar features per closed loop, enabled all four coefficient slots per loop, and replaced the attached loop stack with site-`x` based Wilson loops covered by covariance/Jacobian tests.
- Added the gauge-covariance derivation showing why non-Abelian loop stacks used in `Delta_{x,mu}` must be based at the active link's starting site.
- Clarified that the U(2) CNN coefficient inputs use invariant plaquette/rectangle tensors, while the `Delta` update uses a separate active-site based attached loop stack.
- Cleared stale 2du2 generated gauge configs, model checkpoints, CSV diagnostics, and plots that must be regenerated after the U(2) gauge-convention and field-transform changes.
- Added a U(1)-to-U(2) beta-normalization derivation showing the determinant-sector estimate `beta_U2 ~= 4 beta_U1` and why the U(2) determinant phase aligns with the U(1) plaquette angle for topology comparisons.
- Filled in the U(1) field-transform documentation with 8-subset updates, CNN input/output channels, phase-shift formulas, and the code-matched Jacobian cos/sin terms.
- Reworked the U(2) gauge-invariant CNN input discussion in `presentation/Field_transform.md` to express `ReTr C`, `ImTr C`, `det C`, and `Tr C^n` in split phase-quaternion variables.
- Expanded the gauge-covariance explanation in `presentation/Field_transform.md` to contrast Abelian U(1) scalar loop features with non-Abelian U(2) traceless color components.
- Added the code-matched U(1) and U(2) Wilson action comparison to `presentation/Field_transform.md`, including the split U(2) plaquette trace factor.
- Added the current 2du1 and 2du2 topological-charge definitions to `presentation/Field_transform.md`.
- Documented the current 2du2 split U(2) representation as a central U(1) phase plus SU(2) quaternion, matching the code's group operations and algebra layout.
- Expanded the old U(2) base gauge-covariance discussion with the explicit `loop_sin_cos_features` channel structure and how the full field-transform layout used central and traceless loop components.
- Clarified the old U(2) base discussion by separating CNN input loop features from CNN output coefficient slots.

## 2026-05-04

- Converted the U(2) base transform and base CNN to a scalar-only gauge-symmetric diagnostic path, corrected the U(2) rectangle loop multiplication order for non-Abelian gauge invariance, removed U(2) debug regularizer CLI/loss terms, and added gauge-covariance/logdet-invariance tests.
- Expanded `presentation/Field_transform.md` with the current U(2) scalar-only channel layout, the repository gauge-transform convention, and why gauge-symmetry breaking can destabilize transformed-force training.
- Matched the U(2) scalar-only base model more closely to U(1) by zeroing the cos-like phase coefficient slots and keeping only sin-like phase outputs.
- Added a `2du1/model_training/sub.sh` PBS training script mirroring the current 2du2 submission setup with U(1)-supported CLI arguments.
- Corrected `presentation/Field_transform.md` to separate U(2) invertibility from gauge covariance, note that ordinary CNNs over traceless color channels do not guarantee gauge symmetry, and outline a scalar-input/covariant-basis U(2) design.
- Added U(2)-only configurable force-loss weights for the L2/L4/L6/L8 transformed-force objective, preserving the default equal-weight objective.
- Extended `2du2/model_training/train.py` with `--loss_weights` and `--delta_reg` overrides for beta=8 high-order force-tail experiments.
- Prepared the `2du2` beta=8 training entry point to accept tail-weighted and L6/L8-only ablation runs without changing the default source behavior.
- Expanded U(2) training diagnostics to report resolved loss weights, weighted force loss, and the realized delta-regularization penalty alongside the raw force components.

## 2026-05-03

- Tightened U(2) default field-transformation training (lower base LR, light weight decay, `ReduceLROnPlateau` patience 1) and added global gradient clipping after the shared backward in U(1)/U(2) `FieldTransformation.train_step`.
- Extended `2du1` and `2du2` `model_training/train.py` with optional `--max_grad_norm`, `--plateau_factor`, and `--plateau_patience`, and log the merged `hyperparams` after construction.
- Added optional `return_diagnostics` on U(1)/U(2) `inverse`, per-epoch `inverse_diag` lines (rank 0) after each training epoch using a small fixed test slice, and 1-based epoch indices on U(1)/U(2) training loss plots.
- Made U(2) training loss DDP-global before logging/checkpointing/scheduling, added per-epoch LR logging, and added U(2)-only validation early stopping with CLI override.
- Added U(2)-only per-epoch training diagnostics for pre-clip gradient norm, model parameter norm, transform update size, coefficient saturation, Jacobian logdet statistics, and action/Jacobian/total force-loss components.
- Added a U(2)-only delta-size regularization term to the training loss to discourage runaway transformation/Jacobian growth while preserving the L2/L4/L6/L8 force objective.

## 2026-04-30

- Reconfigured the U(2) scaling driver for the beta=4.0 training point, beta=4.0-8.0 evaluation grid, seed 1029, and overwrite-by-default reruns.
- Removed old U(2) beta=3.0 scaling artifacts while preserving unrelated beta=3.0 exploratory checkpoints.
- Regenerated the U(2) L8 and L16 beta=4.0 gauge ensembles and retrained the corresponding base scaling checkpoints.
- Refreshed the full 2048-sample standard HMC evaluations for U(2) L8/L16 beta=4.0-8.0 and verified acceptance rates in the requested range.
- Updated the U(2) scaling notebook constants to the beta=4.0 training/evaluation presentation target; full notebook execution remains blocked until the beta=4.0 FT-HMC grid is available.
- Measured the current U(2) FT-HMC path at roughly 30 minutes for a 128-sample L8 probe, making the requested full 2048-sample L8/L16 beta=4.0-8.0 FT-HMC refresh impractical in the local single-GPU run.
- Split U(1) and U(2) field-transform compile handling into evaluation-only explicit paths, with regular training defaults and separate force-only compiled callables for FT-HMC.
- Tightened the U(1)/U(2) compile split so regular field transformations stay eager after evaluation compile is enabled.

## 2026-04-29

- Replaced the U(1) base field-transformation module with the optimized sin/cos implementation including rectangle-loop terms.
- Upgraded the U(2) base field transformation to non-volume-preserving plaquette/rectangle loop terms with exact active-link local Jacobians.
- Removed the duplicate U(1) optimized training entry points now covered by the unified training script.
- Retrained the U(1) beta=3.0 L8 and L16 scaling checkpoints with the unified field-transformation module.
- Documented the U(2) Jacobian-check and optional compile controls for the upgraded base transformation.

## 2026-04-28

- Added the U(2) base field-transformation, model-training, and FT-HMC evaluation pipeline.
- Added U(2) matrix-to-split conversion, batched plaquette/action helpers, and focused tests for the new FT-HMC path.
- Updated U(2) documentation to describe training and evaluation alongside gauge generation.
- Added standard HMC evaluation variants under `2du1/evaluation/hmc` and `2du2/evaluation/hmc`.
- Added a U(1) scaling pre-run driver and an analysis-only notebook for HMC versus FT-HMC scaling metrics.
- Added shared plotting and resampling helpers for notebook analysis.
- Renamed the result notebook area to `presentation` and the U(1) workflow script area to `2du1/scripts`.
- Updated the U(1) scaling driver to reuse existing 2048-config training ensembles and completed 16-epoch models, and to use hand-probed no-tune evaluation step sizes for local presentation runs.
- Added lattice-specific FT-HMC scaling step-size defaults after short no-tune acceptance probes.
- Completed the U(1) scaling presentation run with 2048-sample HMC/FT-HMC evaluations and handled frozen-HMC infinite ratio points in the analysis notebook.
- Added a U(2) base scaling driver with resumable gauge, training, accept-rate-checked evaluation stages, and a U(2) scaling presentation notebook.
- Tuned U(2) base scaling gauge generation and no-tune evaluation step-size defaults with short CUDA acceptance probes.
- Fixed U(2) base-training loss normalization to average per configuration, use sample-weighted epoch means, and use more conservative optimizer defaults after CUDA loss probes.
- Fixed U(1) base and optimized training loss normalization to average per configuration and use sample-weighted epoch means.
- Retrained the U(1) L8 and L16 base scaling checkpoints with the corrected loss normalization.

## 2026-04-27

- Initialized the repository scaffold for neural transformed HMC research.
- Added required project initialization documents from `INIT.md`.
- Set up symmetric `2du1` and `2du2` workspaces while keeping U(2) as a structural placeholder.
- Reproduced the U(1) gauge generation, neural field-transformation training, and FT-HMC evaluation pipeline.
- Reorganized reusable Python code into `nthmc.core`, `nthmc.u1`, and `nthmc.u2`, with editable-install metadata in `pyproject.toml`.
- Added the U(1) add-cos optimized transformation path and corresponding model-training/evaluation entry points.
- Kept generated arrays, checkpoints, logs, plots, and diagnostics out of source-oriented directories with `.gitignore` coverage.
- Replaced the `2du2` SU(2) placeholder with U(2) complex-matrix standard HMC gauge generation, diagnostics, docs, and focused tests.
- Changed U(2) gauge generation internals to a split U(1) phase plus SU(2) quaternion representation while preserving complex-matrix config output.
