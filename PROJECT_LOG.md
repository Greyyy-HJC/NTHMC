# PROJECT_LOG.md

This is an append-only development history for NTHMC.

## 2026-05-27

- Reworked U(2) observable autocorrelation estimates to follow the U(1)-style form with smaller statistical uncertainties, and started evaluation jobs to compare the revised uncertainty behavior.
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
