# U(2) L32 Full-Pipeline Benchmark

This directory archives the final evidence for the U(2) L32, beta 10 full-pipeline comparison.
The historical pipeline uses the pre-migration PyTorch runtime; the current pipeline uses JAX
for gauge generation, HMC, and FT-HMC evaluation while model training remains PyTorch.

The comparison uses one seed (`1029`) and 4096-configuration HMC/FT-HMC evaluations. Each
pipeline uses its own trained base model. Raw final logs live in `logs/`, final acceptance and
topology samples live in `dumps/`, and run provenance lives in `metadata/`.

Run `python summarize.py` to regenerate `summary.md` and `timings.csv`. Large gauge arrays,
model checkpoints, probe jobs, failed runs, and PBS submission scripts are intentionally not
part of this result-reproducible archive.
