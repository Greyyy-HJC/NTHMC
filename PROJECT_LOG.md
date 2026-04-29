# PROJECT_LOG.md

This is an append-only development history for NTHMC.

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
