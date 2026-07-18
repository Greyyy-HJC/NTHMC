# U(1) L32 Full-Pipeline Benchmark

Historical source: `e3847a90570f2434117177c80872c9306b69a93b`. Current evaluation uses JAX while training remains PyTorch.
FT-HMC uses each pipeline's own trained model; this is not a same-checkpoint backend microbenchmark.
Historical FT-HMC execution mode: **compiled force path**.
Each implementation was independently tuned to the same acceptance target before the 4096-configuration run; step sizes therefore need not match.
Stage estimates cover one training seed followed by one HMC and one FT-HMC evaluation; queue waits, probes, and optional tuning are excluded.

## Stage Total Estimates

| Stage | Timing basis | Old PyTorch | New | Speedup |
|---|---|---:|---:|---:|
| gauge generation | single-job PBS walltime | 1867.00 s (0:31:07.00) | 131.00 s (0:02:11.00) | 14.252 |
| model training | single-seed reported elapsed | 362.00 s (0:06:02.00) | 384.00 s (0:06:24.00) | 0.943 |
| HMC evaluation | reported sampler total | 63.83 s (0:01:03.83) | 18.93 s (0:00:18.93) | 3.372 |
| FT-HMC evaluation | reported sampler total | 2318.67 s (0:38:38.67) | 1989.62 s (0:33:09.62) | 1.165 |
| **sequential pipeline** | sum of stage totals | **4611.50 s (1:16:51.50)** | **2523.55 s (0:42:03.55)** | **1.827** |

## Evaluation Parameters

| Sampler | Old PyTorch step size | New JAX step size |
|---|---:|---:|
| HMC | 0.280 | 0.280 |
| FT-HMC | 0.300 | 0.300 |

## Timing Details

| Metric | Old PyTorch (s) | New (s) | Speedup |
|---|---:|---:|---:|
| gauge_pbs | 1867.00 | 131.00 | 14.252 |
| training_seed1029 | 362.00 | 384.00 | 0.943 |
| hmc_thermalization | 2.03 | 2.75 | 0.738 |
| hmc_run | 61.80 | 16.18 | 3.820 |
| hmc_reported_total | 63.83 | 18.93 | 3.372 |
| hmc_pbs | 112.00 | 54.00 | 2.074 |
| fthmc_model_load | 0.21 | 0.14 | 1.500 |
| fthmc_thermalization_compile | 293.09 | 24.95 | 11.747 |
| fthmc_run | 2025.38 | 1964.52 | 1.031 |
| fthmc_reported_total | 2318.67 | 1989.62 | 1.165 |
| fthmc_pbs | 2449.00 | 2059.00 | 1.189 |

## Evaluation Throughput

| Sampler | Old configs/s | New configs/s |
|---|---:|---:|
| HMC | 66.278 | 253.152 |
| FT-HMC | 2.022 | 2.085 |

## Sanity Checks

| Sampler | Implementation | Acceptance | Mean plaquette | Topology mean | Topology std | Samples |
|---|---|---:|---:|---:|---:|---:|
| HMC | old PyTorch | 0.7139 | 0.863737 | -0.0796 | 3.0000 | 4096 |
| HMC | new JAX | 0.7224 | 0.863119 | -0.0486 | 2.7173 | 4096 |
| FT-HMC | old PyTorch | 0.6763 | 0.863897 | -0.0366 | 2.8081 | 4096 |
| FT-HMC | new JAX | 0.7070 | 0.863786 | -0.1345 | 2.9450 | 4096 |
