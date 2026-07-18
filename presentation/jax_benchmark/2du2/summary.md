# U(2) L32 Full-Pipeline Benchmark

This benchmark compares the archived PyTorch L32 beta10 pipeline with the current JAX runtime and PyTorch training pipeline.
FT-HMC uses each pipeline's own trained model, so this is a full-pipeline comparison rather than a same-checkpoint microbenchmark.
Stage estimates cover one training seed followed by one HMC and one FT-HMC evaluation; queue waits, probes, and optional tuning are excluded.

## Stage Total Estimates

| Stage | Timing basis | Old PyTorch | New | Speedup |
|---|---|---:|---:|---:|
| gauge generation | single-job PBS walltime | 21157.00 s (5:52:37.00) | 495.00 s (0:08:15.00) | 42.741 |
| model training | single-seed reported elapsed | 25894.00 s (7:11:34.00) | 21882.00 s (6:04:42.00) | 1.183 |
| HMC evaluation | reported sampler total | 1496.52 s (0:24:56.52) | 178.12 s (0:02:58.12) | 8.402 |
| FT-HMC evaluation | reported sampler total | 67108.41 s (18:38:28.41) | 24260.02 s (6:44:20.02) | 2.766 |
| **sequential pipeline** | sum of stage totals | **115655.93 s (32:07:35.93)** | **46815.14 s (13:00:15.14)** | **2.470** |

## Evaluation Parameters

| Sampler | Old PyTorch step size | New JAX step size |
|---|---:|---:|
| HMC | 0.200 | 0.200 |
| FT-HMC | 0.080 | 0.080 |

## Timing Details

| Metric | Old PyTorch (s) | New (s) | Speedup |
|---|---:|---:|---:|
| gauge_pbs | 21157.00 | 495.00 | 42.741 |
| training_pbs | 25989.00 | 21991.00 | 1.182 |
| training_reported_total | 25894.00 | 21882.00 | 1.183 |
| training_per_step | 23.45 | 19.82 | 1.183 |
| hmc_thermalization | 674.35 | 12.69 | 53.140 |
| hmc_run | 822.17 | 165.43 | 4.970 |
| hmc_reported_total | 1496.52 | 178.12 | 8.402 |
| hmc_pbs | pending | 206.00 | pending |
| fthmc_model_load | 0.26 | 0.11 | 2.364 |
| fthmc_thermalization_compile | 7268.34 | 1583.69 | 4.589 |
| fthmc_run | 59839.82 | 22676.22 | 2.639 |
| fthmc_reported_total | 67108.41 | 24260.02 | 2.766 |
| fthmc_pbs | 67182.00 | 24589.00 | 2.732 |

## Evaluation Throughput

| Sampler | Old configs/s | New configs/s |
|---|---:|---:|
| HMC | 4.982 | 24.760 |
| FT-HMC | 0.068 | 0.181 |

## Sanity Checks

| Sampler | Implementation | Acceptance | Mean plaquette | Topology mean | Topology std | Samples |
|---|---|---:|---:|---:|---:|---:|
| HMC | old PyTorch | 0.7534 | 0.791215 | 0.2693 | 3.4814 | 4096 |
| HMC | new JAX | 0.0544 | 0.790653 | -1.6143 | 3.6892 | 4096 |
| FT-HMC | old PyTorch | 0.8140 | 0.790874 | -0.0493 | 3.5191 | 4096 |
| FT-HMC | new JAX | 0.8237 | 0.790804 | 0.0049 | 3.5335 | 4096 |

**Caveat:** the archived new-JAX HMC run has acceptance below 0.4. Its timing and the aggregate pipeline estimate describe the recorded run, but they are not an acceptance-matched performance baseline.
