# U(1) L32 Full-Pipeline Benchmark

This directory archives the final evidence comparing the historical PyTorch pipeline with the current JAX runtime and PyTorch training pipeline.

- Historical source: commit `e3847a90570f2434117177c80872c9306b69a93b`.
- Current source: the July 14, 2026 JAX benchmark worktree.
- Common evaluation parameters: `L=32`, `beta=train_beta=4.0`, `n_configs=4096`, `n_steps=10`, `n_thermalization=10`, and seed `1029`.
- Step sizes were independently selected for each implementation and sampler from short probes targeting comparable acceptance; automatic in-run tuning is disabled for the timed 4096-configuration runs.
- Historical FT-HMC uses the compiled force path with a cold, isolated Inductor cache. Its PBS wrapper falls back to eager only if compilation fails and records that fallback in the log.
- FT-HMC uses each pipeline's own trained model, so this is a full-pipeline comparison rather than a same-checkpoint backend microbenchmark.

Raw final logs live in `logs/`, final acceptance and topology samples live in `dumps/`, and run provenance lives in `metadata/`.
Run `python summarize.py` to regenerate `summary.md` and `timings.csv`.

Large gauge arrays, model checkpoints, probe jobs, untuned runs, failed runs, and PBS submission scripts are intentionally not part of this result-reproducible archive.
