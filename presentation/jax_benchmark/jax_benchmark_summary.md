# JAX Full-Pipeline Benchmark Summary

This archive compares the historical PyTorch runtime with the current hybrid pipeline:
JAX handles gauge generation, HMC, and FT-HMC evaluation, while model training remains
PyTorch. Each FT-HMC result uses the model trained by its own pipeline.

## Stage Estimates

| System | Pipeline | Gauge generation | Model training | HMC evaluation | FT-HMC evaluation | Sequential total |
|---|---|---:|---:|---:|---:|---:|
| U(1), L32, beta 4 | old PyTorch | 0:31:07.00 | 0:06:02.00 | 0:01:03.83 | 0:38:38.67 | **1:16:51.50** |
| U(1), L32, beta 4 | current | 0:02:11.00 | 0:06:24.00 | 0:00:18.93 | 0:33:09.62 | **0:42:03.55** |
| U(2), L32, beta 10 | old PyTorch | 5:52:37.00 | 7:11:34.00 | 0:24:56.52 | 18:38:28.41 | **32:07:35.93** |
| U(2), L32, beta 10 | current | 0:08:15.00 | 6:04:42.00 | 0:02:58.12 | 6:44:20.02 | **13:00:15.14** |

The sequential estimate covers one training seed, one HMC evaluation, and one FT-HMC
evaluation. It excludes queue waits, step-size probes, optional tuning, and failed runs.
Gauge generation uses single-job PBS walltime, training uses single-seed reported elapsed,
and sampler stages use their reported totals.

The U(1) recorded pipeline total is 1.827x faster and the U(2) recorded total is 2.470x
faster. The U(2) new-JAX HMC run has acceptance 0.0544, so its HMC timing—and therefore the
aggregate U(2) pipeline speedup—is not an acceptance-matched performance baseline. The U(2)
FT-HMC comparison is acceptance-matched within the archived results.

Detailed timings, throughput, acceptance, plaquette, and topology checks are available in
[`2du1/summary.md`](2du1/summary.md) and [`2du2/summary.md`](2du2/summary.md). Run each
directory's `summarize.py` to regenerate its summary and timing CSV from the archived logs
and dumps.
