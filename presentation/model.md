# U(2) model architecture notes

This note summarizes the beta=10.0, L=32, seed=1029 model architecture tests in
`2du2/model_training`. The current source keeps only the baseline and the variants
that are useful for ongoing comparison.

## Current retained tags

| tag | role | architectural change from `base` |
| --- | --- | --- |
| `base` | reference | Two-layer local CNN, hidden width 12, original coefficient caps. |
| `wide` | ablation | Same forward structure as `base`, but hidden width 32. |
| `cap` | ablation | Same backbone as `base`, but output caps use 90% of a 50/50 plaquette/rectangle budget split. |
| `mscap` | candidate | Wide multi-scale feature branches, split plaquette/rectangle heads, and the same 90% 50/50 cap split. |

## Feature definitions

### cap

`cap` changes only the output coefficient bounds. The original `base` caps are:

```text
c_plaq = 1 / 5
c_rect = 1 / 40
```

Under the reversibility budget `4*c_plaq + 8*c_rect <= 1`, this is an 80/20
plaquette/rectangle allocation. The retained cap variants instead use 90% of a
50/50 allocation:

```text
c_plaq = 0.1125
c_rect = 0.05625
```

This uses `4*0.1125 + 8*0.05625 = 0.9` of the total budget and keeps plaquette
and rectangle contributions balanced.

### wide

`wide` changes only the hidden width from 12 to 32. It inherits the base forward
path and the original base coefficient caps.

### split

`split` was tested as a base-width shared trunk with separate plaquette and
rectangle output heads. It did not improve the loss and is not retained in
`choose_model`.

### multi-scale

`mscap` uses three feature branches after a shared input convolution:

```text
3x3 local branch
dilated 3x3 branch
1x1 pointwise branch
```

The branch outputs are concatenated, mixed with a 1x1 convolution, then sent to
separate plaquette and rectangle heads. The output caps are the same 90% 50/50
caps used by `cap`.

### dual

`duocap` was tested as a wide cap model with separate plaquette and rectangle
trunks after a shared input layer. It improved over `base`, but was weaker than
`cap` and `mscap`, so it is not retained.

### corr

`corrcap` was tested as a base-width cap model with a small correction path
added to the base-like logits. It improved over `base`, but was weaker than
`cap` and `mscap`, so it is not retained.

## Seed 1029 training results

All results below use beta=10.0, L=32, seed=1029, and 16 training epochs.

| tag | best test loss | best epoch | last test loss | retained |
| --- | ---: | ---: | ---: | --- |
| `mscap` | 28.263115 | 14 | 28.266971 | yes |
| `widehalf90` / old `widesplitcap` | 28.311005 | 16 | 28.311005 | no, superseded by `mscap` |
| `cap` | 29.376594 | 16 | 29.376594 | yes |
| `corrcap` | 29.442378 | 16 | 29.442378 | no |
| `duocap` | 29.831238 | 15 | 29.898874 | no |
| `base` | 30.295583 | 15 | 31.579908 | yes, reference |
| `wide` | 30.985438 | 16 | 30.985438 | yes, ablation |
| `split` | 33.155722 | 15 | 33.157925 | no |

The main empirical conclusions are:

- The cap change is the dominant single feature.
- Widening alone does not improve the base model.
- Split heads alone are harmful in the base-width setting.
- Multi-scale features combined with the stable cap are currently the best tested direction.
