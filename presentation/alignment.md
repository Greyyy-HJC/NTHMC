# Why the Alignment Term Did Not Help

This note summarizes the U(2) topology-alignment loss tests and why the
alignment term was removed from the training objective.

## Idea

The intended loss was:

```text
force loss - mean(cos(force, grad Q_soft)^2)
```

where `Q_soft` is a differentiable proxy for topology:

```text
Q_soft = sum(sin(theta) + 0.3 sin(2 theta)) / (2 pi)
```

The motivation was reasonable: encourage the learned transformation force to
have a component along the soft-topology gradient, hoping this would help move
the transformed field in topology-changing directions.

## What the tests measured

The relevant test scripts are:

- `tests/test_u2_topology_alignment_diagnostic.py`
- `tests/test_u2_alignment_loss_ablation.py`

The diagnostic script measures the cosine between the transformed force and
`grad Q_soft`. For the beta=10.0, L=32, seed=1029 base checkpoint, the logged
full-force cosine was small and sign-changing:

```text
force_topo_cos:     mean=1.24e-02, std=4.40e-02, min=-5.99e-02, max=7.66e-02
force_topo_abs_cos: mean=3.71e-02
```

The phase-only cosine was slightly larger but still small:

```text
phase_force_topo_abs_cos: mean=4.23e-02
```

The ablation script then compared short training runs from the same checkpoint
using `force_only`, `cos_sq`, `smooth_abs`, and `hinge` alignment variants. In
the 20-step run, `force_only` improved the eval weighted force loss by about:

```text
delta_eval_force_weighted_loss = -9.53e-02
```

The alignment variants did not give a robust improvement over this baseline.
Low or moderate weights were essentially tied with force-only, while large
weights often reduced the force-loss improvement and produced much larger
gradient norms. For example:

| variant | weight | delta eval force loss |
| --- | ---: | ---: |
| force_only | 0 | -9.53e-02 |
| cos_sq | 10 | -9.51e-02 |
| cos_sq | 100 | -8.49e-02 |
| cos_sq | 300 | -6.04e-02 |
| smooth_abs | 10 | -9.48e-02 |
| smooth_abs | 300 | -6.35e-02 |
| hinge | 10 | -9.63e-02 |
| hinge | 300 | -4.31e-02 |

## Why it did not help training

The main issue is that the alignment term optimizes a direction proxy, not the
actual training objective. The training objective is the weighted transformed
force norm, using L2/L4/L6/L8 components. A cosine reward is scale-invariant: it
can improve the angle between `force` and `grad Q_soft` without reducing the
force magnitude. In the worst case, it spends gradient budget rotating or
amplifying the force instead of shrinking it.

The proxy is also not the measured target. `Q_soft` is a smooth plaquette-phase
surrogate, while the desired physical outcome is better topological motion in
the eventual HMC chain. A larger local cosine with `grad Q_soft` does not
guarantee lower transformed-force loss, better acceptance, or lower topology
autocorrelation.

The tested sign-insensitive losses, such as `-cos^2` and smooth `-|cos|`, reward
alignment and anti-alignment equally. That is useful if the only goal is a
nonzero topology-gradient projection, but it also makes the loss less tied to a
specific update direction. The hinge variant avoids rewarding already-large
cosines, but the diagnostic showed most samples were inside the hinge margin, so
it mostly behaved like another competing objective.

Empirically, the baseline force was already nearly orthogonal to `grad Q_soft`,
with absolute cosines around a few percent. Increasing the alignment weight made
the auxiliary term numerically important, but the short ablations did not show a
matching improvement in eval force loss. Large weights instead increased
gradient norms and weakened the force-loss gain.

## Current status

The training loss in `src/nthmc/u2/field_transform.py` is now force-only again.
The old alignment path is left commented near `loss_fn` for quick rollback, and
the topology-gradient helper remains available for diagnostics.

`src/nthmc/u2/field_transform_diagnostics.py` still reports `force_topo_cos`.
That is the useful role for this quantity: observability, not a default training
objective.
