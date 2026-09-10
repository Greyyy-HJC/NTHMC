# U(1) Large-Angle Edge-Link Weighting

This note records the 2du1 experiment that up-weights gauge links whose angles
sit near the regularize boundary \(\pm\pi\). Topology tunneling is mainly due
to those large-angle links; they become rarer as topology freezes, so the
training loss concentrates on them. The architecture remains production
`base`. Only the PyTorch training objective changes. JAX FT-HMC evaluation is
unchanged.

The comparison workspace is archived locally under gitignored
[`temp/2du1/evaluation/edge/`](../../temp/2du1/evaluation/edge/).
Production [`2du1/evaluation/base/`](../../2du1/evaluation/base/) dumps at
\(\beta=3.0\) were not overwritten. The result notebook is
[`temp/presentation/2du1_edge_weight.ipynb`](../../temp/presentation/2du1_edge_weight.ipynb);
plots are under `temp/presentation/plots/`.

## Objective

Physical configurations \(\theta\) enter the usual inverse, \(z=F^{-1}(\theta)\),
and the transformed force \(F_z=\nabla_z S_{\mathrm{eff}}\). Per-link weights
are computed from the **physical** field \(\theta\) and multiplied onto the
matching \(z\)-space force components:

```text
w(θ) = 1 + α * ((1 - cos θ)/2)^p          # smooth, default when δ=0
w(θ) = 1 + α  if |wrap(θ)| ≥ π - δ else 1 # hard threshold when δ>0
loss = ||w ⊙ F_z||_2 / √V + ||w ⊙ F_z||_4 / V^{1/4}
     + ||w ⊙ F_z||_6 / V^{1/6} + ||w ⊙ F_z||_8 / V^{1/8}
```

`wrap` is the existing map onto \([-\pi,\pi)\). Smooth weights are already
\(2\pi\)-periodic and peak at \(\pm\pi\). \(\alpha=0\) recovers the production
force-only loss exactly. Weights are detached, so they do not back-propagate
through \(\theta\).

Training logs both the weighted objective and the unweighted force loss.
Checkpoint selection and the plateau scheduler follow the weighted validation
objective. The unweighted validation force loss is the FT-HMC promotion gate.

CLI: `--edge_weight` (\(\alpha\), default 0), `--edge_power` (\(p\), default 2),
`--edge_delta` (\(\delta\), default 0 = smooth).

This experiment is U(1)-only. U(2) links are matrices and do not have the same
\(\pm\pi\) angle variable.

## Test protocol

Train at L=16, \(\beta=3.0\), same 4-GPU protocol as production base
(`lr=0.001`, 16 epochs, batch 64, `max_grad_norm=5`). Evaluate FT-HMC with the
same checkpoint at \(\beta\in\{3.0,4.0,5.0\}\). Extra HMC and production-base
FT-HMC chains at \(\beta=4.0\) and \(5.0\) are generated for the comparison;
they start from random and thermalize, so no new training ensembles are needed.

A candidate advances to multi-seed FT-HMC only if:

1. unweighted validation force loss is within about 5–10% of the seed-1029
   production `base` control;
2. inverse convergence remains healthy;
3. a short FT step-size probe can be retuned into acceptance \([0.55,0.85]\).

Promoted candidates are scored with `R_gamma(16)` and `R_deltaQ` versus matched
HMC and versus production `base`, using `autocorrelation_from_chi` as in
[`presentation/2du1_scaling.ipynb`](../2du1_scaling.ipynb). Win rule at
\(\beta=3.0\): both metric means beat `base`, and at least one is \(1\sigma\)
separated. No topology benefit is claimed until the eight-seed confirmation
completes.

## L16 beta=3 seed=1029 weight screen

| RUN_TAG | form | parameters | status |
|---|---|---|---|
| `base` | none | production control, not retrained | existing checkpoint |
| `edge_s2_w1` | smooth | \(p=2\), \(\alpha=1\) | T1 FT-HMC complete (resubmit 175744) |
| `edge_s2_w3` | smooth | \(p=2\), \(\alpha=3\) | T1 FT-HMC complete (resubmit 175745) |
| `edge_s2_w10` | smooth | \(p=2\), \(\alpha=10\) | T1 FT-HMC complete (resubmit 175746) |
| `edge_h05_w3` | hard | \(\delta=0.05\), \(\alpha=3\) | T1 FT-HMC complete (175736) |
| `edge_h15_w3` | hard | \(\delta=0.15\), \(\alpha=3\) | T1 FT-HMC complete (175737); ranking winner |

Unweighted validation force loss at the best-objective epoch versus production
`base` seed 1029 (`10.525`). All five stay within 2% of the control (gate was
5–10%). Inverse iterations stayed at 3–9 with round-trip error \(<10^{-6}\).

| RUN_TAG | best epoch | best val force loss | vs control | mean weight |
|---|---:|---:|---:|---:|
| `base` | 16 | 10.525 | — | 1 |
| `edge_s2_w1` | 16 | 10.448 | −0.7% | 1.37 |
| `edge_s2_w3` | 16 | 10.394 | −1.2% | 2.10 |
| `edge_s2_w10` | 16 | 10.362 | −1.6% | 4.66 |
| `edge_h05_w3` | 16 | 10.635 | +1.0% | 1.05 |
| `edge_h15_w3` | 16 | 10.552 | +0.3% | 1.15 |

FT-HMC screen at \(\beta=3.0\), seed 1029, 512 configs, automatic step tuning
starting from `0.35`. Metrics use the first 512 samples of matched HMC and
production `base` chains. All five candidates stayed at step 0.35 with
acceptance in \([0.55,0.85]\). Smooth screens were recovered after JAX OOM
resubmits 175744–175746.

| RUN_TAG | accept | step | \(R_\gamma\) vs HMC | \(R_{\Delta Q}\) vs HMC | \(R_\gamma\) vs base | \(R_{\Delta Q}\) vs base |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 0.819 | 0.35 | 1.219 | 1.504 | — | — |
| `edge_s2_w1` | 0.809 | 0.35 | 1.356 | 1.739 | 1.113 | 1.156 |
| `edge_s2_w3` | 0.793 | 0.35 | 1.485 | 1.683 | 1.218 | 1.119 |
| `edge_s2_w10` | 0.807 | 0.35 | 1.218 | 1.695 | 0.999 | 1.127 |
| `edge_h05_w3` | 0.736 | 0.35 | 1.378 | 1.560 | 1.130 | 1.037 |
| `edge_h15_w3` | 0.756 | 0.35 | 1.740 | 1.669 | 1.427 | 1.110 |

Versus production `base` on this seed, `edge_h15_w3` is \(+43\%\) in
\(R_\gamma(16)\) and \(+11\%\) in \(R_{\Delta Q}\). Next is `edge_s2_w3`
(\(+22\%\), \(+12\%\)). `edge_s2_w10` does not beat `base` on \(R_\gamma\).
This ranking did not survive eight-seed 2048-config confirmation.

## Eight-seed FT-HMC at \(\beta=3.0\) (2048 configs)

All acceptances sit in \([0.55,0.85]\) at step 0.35. Jackknife is over healthy
seeds versus matched HMC and production `base`.

| tag | seeds | vs HMC \(R_\gamma\) | vs HMC \(R_{\Delta Q}\) | vs `base` \(R_\gamma\) | vs `base` \(R_{\Delta Q}\) |
|---|---:|---:|---:|---:|---:|
| production `base` | 8 | 1.073(24) | 1.511(13) | — | — |
| `edge_s2_w1` | 8 | 1.076(28) | 1.536(16) | 1.003(10) | 1.017(10) |
| `edge_s2_w3` | 8 | 1.065(41) | 1.519(14) | 0.991(21) | 1.005(9) |
| `edge_s2_w10` | 8 | 1.083(40) | 1.544(20) | 1.009(28) | 1.022(12) |
| `edge_h05_w3` | 8 | 1.068(21) | 1.419(17) | 0.998(27) | 0.939(9) |
| `edge_h15_w3` | 8 | 1.113(19) | 1.453(18) | 1.039(15) | 0.962(11) |

No setting produces a useful topology gain versus `base`. Smooth weights are
consistent with `base` (\(R_\gamma\) within \(1\sigma\) of 1; \(R_{\Delta Q}\)
excesses of \(0\)–\(2\%\)). `h15` has a \(\sim 2.6\sigma\) \(R_\gamma\) excess
but \(R_{\Delta Q}\) is worse at \(\sim 3.5\sigma\). `h05` is worse on
\(R_{\Delta Q}\) at \(\sim 6.7\sigma\). Seed-to-seed scatter is a few percent;
T1 512-config swings of \(+11\%\)–\(+43\%\) were chain-length noise. Relative
to HMC, the \(\sim R_\gamma=1.07\), \(R_{\Delta Q}=1.5\) gain is the existing
production `base` result, not extra from weighting.

## Confirmation protocol

Single-seed T1 ranking is treated as noisy. All five T1 tags reuse seed 1029
and train the remaining seven seeds with the same 4-GPU protocol. After
checkpoints exist, evaluate eight seeds at \(\beta=3.0\) first (2048 configs,
per-seed step retune into \([0.55,0.85]\)). Jackknife over seeds whose
edge-model acceptance sits in that band, versus HMC and versus production
`base` at the same \((\beta,\mathrm{seed})\). Extra betas \(4.0\) and \(5.0\)
follow for tags that remain competitive at \(\beta=3.0\). Win rule at
\(\beta=3.0\): both metric means beat `base`, and at least one is \(1\sigma\)
separated.

## Multi-seed / multi-beta status

HMC at \(\beta=4.0\) and \(5.0\) is complete for all eight seeds (2048 configs).
Production-base FT-HMC at those betas used automatic step tuning from 0.35.

All five T1 tags have eight training checkpoints and complete β=3 2048-config
FT-HMC (8/8). Extra-beta eight-seed FT-HMC was not justified by the win rule.

| tag | 8-seed trains | β=3 FT-HMC (2048) |
|---|---|---|
| `edge_s2_w1` | complete | complete 8/8 |
| `edge_s2_w3` | complete | complete 8/8 |
| `edge_s2_w10` | complete | complete 8/8 |
| `edge_h05_w3` | complete | complete 8/8 |
| `edge_h15_w3` | complete | complete 8/8 |

| beta | HMC | base FT-HMC | edge FT-HMC | notes |
|---:|---|---|---|---|
| 3.0 | complete (production) | complete (production, fixed step 0.35) | all five tags 8/8 | confirmation complete |
| 4.0 | complete; accept ~0.70–0.72, step ~0.306 | complete (8/8; accept 0.74–0.80, step 0.35) | not expanded | no competitive tag at β=3 |
| 5.0 | complete; accept ~0.82–0.83, step ~0.263 (seed 2025: 0.633 at 0.284) | complete (8/8; accept 0.62–0.71, step 0.35) | seed-1029 probes for `s2_w1` / `s2_w3` / `h15` | mixed, single-seed only |

β=5 seed-1029 probe (2048 configs, step 0.35):

| tag | accept | \(R_\gamma\) vs base | \(R_{\Delta Q}\) vs base |
|---|---:|---:|---:|
| `edge_s2_w1` | 0.658 | 1.058 | 1.028 |
| `edge_s2_w3` | 0.688 | 0.903 | 0.935 |
| `edge_h15_w3` | 0.577 | 1.094 | 1.009 |

## Bottom line

Eight-seed β=3 FT-HMC (2048 configs) finds no useful topology gain versus
production `base`. Smooth weights are consistent with `base`. `h15` trades a
small \(R_\gamma\) excess for a significant \(R_{\Delta Q}\) loss. Default
training remains unweighted (`edge_weight=0`). See the archived notebook
[`temp/presentation/2du1_edge_weight.ipynb`](../../temp/presentation/2du1_edge_weight.ipynb).
