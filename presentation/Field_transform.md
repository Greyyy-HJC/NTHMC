# Detailed Comparison: 2D U(1) vs 2D U(2) Field Transformations

This note documents the field transformations implemented in:

- `src/nthmc/u1/field_transform.py`
- `src/nthmc/u2/field_transform.py`
- `src/nthmc/u1/models.py`
- `src/nthmc/u2/models.py`

The key structural distinction is:

- U(1): scalar active-link Jacobian factor.
- U(2): local $4\times 4$ active-link Jacobian block.

---

## 1. Core Principle: Gauge Covariance

Gauge transformation in the convention used by the current U(2) code:
$$
U_{x,\mu} \rightarrow G_{x+\hat\mu} U_{x,\mu} G_x^\dagger .
$$

A field transformation $F$ satisfies:
$$
F(U^G) = F(U)^G .
$$

Transformations are built from local Wilson loops:

- plaquettes
- $1\times2$, $2\times1$ rectangles

For non-Abelian groups, this statement is not enough by itself. A closed
loop matrix based at site $x$ transforms by conjugation:
$$
C_x \rightarrow G_x C_x G_x^\dagger ,
$$
so its trace-like scalar features are gauge invariant, but its traceless
color components are only gauge covariant in the color frame at the loop
base point. Ordinary CNN channels are scalar channels; feeding untransported
traceless color components to an ordinary CNN does not, by itself, preserve
gauge covariance.

Links are split into 8 checkerboard subsets:
$$
(\mu,\ x\bmod 2,\ y\bmod 2).
$$

Each layer updates one subset. Active links in the same subset are independent within that layer.

---

## 2. U(1) Field Transformation

### Update rule

$$
\theta' = \theta + \Delta\theta(\theta)
$$

Each active link is updated independently.

### Jacobian

$$
J = 1 + \frac{\partial \Delta\theta}{\partial\theta}
$$

If
$$
\left|\frac{\partial \Delta\theta}{\partial\theta}\right| < 1
$$
then
$$
J > 0
$$

Thus each layer is strictly monotone and invertible.

---

## 3. U(2) Field Transformation

### Representation

$$
U = (\phi, q), \quad q \in SU(2)
$$

### Update rule

$$
U' = \exp(\Delta(U)) \, U
$$

$$
\Delta(U) \in u(2)
$$

For the left-multiplication update of link $U_{x,\mu}$ to be gauge
covariant in this convention, the algebra update must transform at the
left endpoint $x+\hat\mu$:
$$
\Delta_{x,\mu}(U^G)
=
G_{x+\hat\mu} \Delta_{x,\mu}(U) G_{x+\hat\mu}^\dagger .
$$

### Key structural property

**CNN coefficients depend only on frozen subsets**, not on active links:
$$
\frac{\partial k}{\partial U_{\text{active}}} = 0
$$

Thus Jacobian derivatives only arise from loop features.

### Gauge-covariance issue in the old U(2) base implementation

The old `src/nthmc/u2/field_transform.py` implementation used
`loop_sin_cos_features`, including both trace-like scalar channels and
traceless color-vector channels, as ordinary CNN inputs. It also lets the CNN
output independent coefficients for phase and traceless algebra directions.

This is sufficient for the local Jacobian analysis below, but it does **not**
guarantee U(2) gauge covariance. The issue is that traceless color-vector
features transform by local adjoint rotations, while the CNN treats channel
components as fixed scalar channels. Therefore the old base transform could
learn gauge-frame-dependent updates.

This is the main structural difference from U(1). In U(1), the corresponding
loop angles are gauge-invariant scalars because the group is Abelian.

### Current scalar-only diagnostic implementation

The current U(2) `base` implementation is the minimal gauge-symmetric
diagnostic variant. It keeps the original full coefficient layout at the
field-transform interface, but the CNN itself only sees gauge-invariant
scalar features and only produces phase-update coefficients.

For each loop, `loop_sin_cos_features` has 8 channels:

- channel 0: scalar sin-like phase feature
- channels 1:4: traceless sin-like color features
- channel 4: scalar cos-like phase feature
- channels 5:8: traceless cos-like color features

The scalar-only U(2) base keeps only channels 0 and 4 before the CNN.
Therefore:

- plaquette CNN input has 2 channels
- rectangle CNN input has 4 channels, from 2 rectangle orientations times
  2 scalar channels

The CNN outputs only phase coefficients:

- plaquette: 4 loops times 2 phase slots = 8 nonzero channels
- rectangle: 8 loops times 2 phase slots = 16 nonzero channels

These are expanded back to the full field-transform layout:

- plaquette full layout: 16 channels
- rectangle full layout: 32 channels

For each loop, coefficient slots 0 and 2 are phase slots, while slots 1 and
3 are traceless color slots. The current base model sets slots 1 and 3 to
zero. Thus the total full-layout output has 24 nonzero phase channels and
24 identically zero traceless channels.

This scalar-only update is gauge covariant because the update is proportional
to the central generator $iI$. The scalar coefficient is gauge invariant, and
$iI$ commutes with every local gauge rotation.

The U(2) rectangle loop multiplication order must also be a closed
non-Abelian Wilson loop. In U(1), the additive rectangle angle is insensitive
to ordering, but in U(2) the matrix product order is physical. If the order
does not close as a Wilson loop, even trace-like rectangle features are not
gauge invariant.

The current tests check:
$$
F(U^G)=F(U)^G,
\qquad
\log|\det J(U^G)|=\log|\det J(U)|.
$$

### Why broken gauge symmetry can damage training

The target Wilson action is gauge invariant, so it is constant along each
gauge orbit. A gauge-covariant field transform preserves this structure: the
transformed potential
$$
S(F(U))-\log|\det J(U)|
$$
is also gauge invariant.

If the learned transform is not gauge covariant, the transformed potential can
vary along pure-gauge directions. This creates several training problems:

- The force loss can reward cancellation patterns that depend on an arbitrary
  gauge frame rather than on physical gauge-invariant structure.
- The Jacobian force can acquire components along gauge-orbit directions even
  though the target action has no physical restoring force there.
- Gauge-equivalent configurations can produce different CNN coefficients,
  making the loss noisier and harder to optimize.
- The network can reduce the local training objective by creating gauge-frame
  dependent Jacobian structure, but this need not improve FT-HMC acceptance
  because the proposal then sees artificial gauge-orbit roughness.

This explains why the U(2) loss can behave differently from U(1) under
similar settings. In U(1), loop features are Abelian scalar angles. In U(2),
untransported traceless loop components carry local color-frame information,
so treating them as ordinary scalar CNN channels can break the symmetry that
the HMC target distribution has.

---

## 4. U(2) Jacobian Structure

Introduce tangent perturbation:
$$
U_X = \exp(X)U
$$

Output tangent:
$$
Y = \log(U'_X U'^\dagger)
$$

Jacobian:
$$
J = \frac{\partial Y}{\partial X}
$$

Each active link contributes a $4\times4$ real matrix.

---

## 5. Decomposition of the Jacobian

$$
J = Q + E
$$

where:

- $Q = \mathrm{Ad}_{\exp(\Delta)}$
- $E = D\exp_\Delta[D\Delta]$

Here $D$ means the differential, or first-order linearization, of a map:

- $D\Delta$ maps an input tangent $X$ to the induced first-order change in the algebra update $\Delta(U)$.
- $D\exp_\Delta[\cdot]$ maps that first-order algebra change through the exponential map at the base point $\Delta$.

Thus $E$ is the part of the output tangent caused by the active-link dependence of $\Delta(U)$. It is shorthand for the linear map:
$$
X
\mapsto
D\exp_\Delta\!\left[D\Delta[X]\right].
$$

### Property of $Q$

In the chosen orthonormal basis of $u(2)=u(1)\oplus su(2)$:

- $Q$ is orthogonal
- $\|Q\|_2 = 1$
- $\det Q = 1$

---

## 6. Invertibility Condition

We write:
$$
J = Q(I + Q^{-1}E)
$$

If
$$
\|E\|_2 < 1
$$

then
$$
\|Q^{-1}E\|_2 < 1
$$

and therefore $I + Q^{-1}E$ is invertible via Neumann series:
$$
(I + Q^{-1}E)^{-1} = \sum_{n=0}^{\infty}(-Q^{-1}E)^n
$$

Thus:
$$
\boxed{J \text{ is non-singular}}
$$

---

## 7. Current U(2) Base Caps

The current U(2) `base` model returns:
$$
k_{\rm plaq} = \frac{\tanh z_{\rm plaq}}{5},
\qquad
k_{\rm rect} = \frac{\tanh z_{\rm rect}}{40}.
$$

Since $\tanh z$ is strictly bounded by 1 for finite real $z$:
$$
|k_{\rm plaq}| < \frac{1}{5},
\qquad
|k_{\rm rect}| < \frac{1}{40}.
$$

In the current scalar-only base, these caps apply to the nonzero phase
coefficient channels. The traceless coefficient channels are identically zero,
so they are trivially bounded.

For one U(2) loop, the sin-like and cos-like coefficient groups give the conservative derivative bound:
$$
\|D\Delta_l\|_2 \le 2c_l.
$$

For one active link:

- 2 plaquette loops contribute.
- 4 rectangle loops contribute.

Therefore:
$$
\|E\|_2
\le
4c_{\rm plaq} + 8c_{\rm rect}.
$$

The current caps give:
$$
4c_{\rm plaq}+8c_{\rm rect}
<
4\cdot\frac15
+
8\cdot\frac1{40}
=
1.
$$

Thus the current caps imply $\|E\|_2<1$, which is exactly the sufficient condition used above. This is why the present `base` caps guarantee that every local U(2) active-link Jacobian block is invertible in real arithmetic.

Numerically, the margin is small when `tanh` saturates close to 1, but the mathematical bound is strict because the coefficient caps are strict.

---

## 8. Determinant Sign

Define:
$$
J(t) = Q + tE, \quad t \in [0,1]
$$

Since
$$
\|Q^{-1} tE\|_2 < 1
$$
for all $t$, $J(t)$ is non-singular along the path.

Thus determinant cannot cross zero.

Since:
$$
\det J(0) = \det Q = 1 > 0
$$

we conclude:
$$
\boxed{\det J > 0}
$$

---

## 9. Gauge-Covariant U(2) Design

A gauge-covariant U(2) field transform should separate scalar neural-network
data from color-covariant algebra data.

For each active link $U_{x,\mu}$, use:

1. **Gauge-invariant scalar CNN inputs**

   Examples:
   $$
   \mathrm{ReTr}\,C,\quad \mathrm{Im}\det C,\quad
   \mathrm{Re}\det C,\quad \mathrm{Tr}(C C^\dagger)
   $$
   for plaquette and rectangle loops. The ordinary CNN should only see such
   scalar fields.

2. **Gauge-invariant scalar CNN outputs**

   The network outputs scalar coefficients:
   $$
   a^{(0)}_{x,\mu},\quad a^{(r)}_{x,\mu}.
   $$
   These coefficients are invariant under local gauge rotations.

3. **Gauge-covariant algebra basis elements**

   Build basis matrices $B^{(r)}_{x,\mu}$ from closed loops based at the
   left endpoint $x+\hat\mu$, or from loops parallel transported to
   $x+\hat\mu$.
   Each basis element must transform as:
   $$
   B^{(r)}_{x,\mu}(U^G)
   =
   G_{x+\hat\mu} B^{(r)}_{x,\mu}(U) G_{x+\hat\mu}^\dagger .
   $$

   A standard traceless anti-Hermitian basis contribution is:
   $$
   B(C_x)
   =
   \left[
   \frac{C_x-C_x^\dagger}{2}
   -
   \frac{1}{2}\mathrm{Tr}
   \left(\frac{C_x-C_x^\dagger}{2}\right) I
   \right] .
   $$

Then define:
$$
\Delta_{x,\mu}
=
a^{(0)}_{x,\mu}\, iI
+
\sum_r a^{(r)}_{x,\mu} B^{(r)}_{x,\mu}.
$$

This transform has more freedom than U(1) because it can update SU(2)
traceless directions through the covariant bases $B^{(r)}_{x,\mu}$, but it
still preserves gauge covariance because all CNN inputs and outputs are scalar
gauge invariants.

### Practical minimum diagnostic variants

- **Scalar-only diagnostic:** feed only gauge-invariant scalar loop features to
  the CNN and update only the $iI$ phase direction. This should behave most
  similarly to U(1), and is useful for isolating whether current U(2) loss
  pathologies come from non-covariant color channels.
- **Gauge-covariant U(2) base:** keep scalar CNN inputs, but add traceless
  updates through transported adjoint loop bases as above.

Both variants should be tested with:
$$
F(U^G)=F(U)^G,
\qquad
\log|\det J(U^G)|=\log|\det J(U)|.
$$

---

## 10. Global Invertibility

Each layer updates disjoint subsets → block factorization:

- U(1): scalar factors
- U(2): $4\times4$ blocks

For a U(2) subset layer:
$$
\det J_{\rm layer}
=
\prod_{\ell\in{\rm active}}
\det J_\ell .
$$

Since the current caps make every local block $J_\ell$ non-singular with positive determinant, every subset layer is a local diffeomorphism.

To upgrade this from local to global invertibility, introduce the scaled layer:
$$
F_{i,t}(U)=\exp(t\Delta_i(U))U,
\qquad
t\in[0,1].
$$

At $t=0$, this is the identity map. For every $t\in[0,1]$, the perturbation bound scales as:
$$
\|E_t\|_2
\le
t(4c_{\rm plaq}+8c_{\rm rect})
<1.
$$

Therefore every $F_{i,t}$ is a local diffeomorphism along a continuous homotopy from the identity to the actual subset layer $F_{i,1}$.

The full U(2) lattice field lives on a finite product of compact connected U(2) manifolds. A local diffeomorphism on this compact connected manifold is a covering map. Since $F_{i,1}$ is homotopic to the identity through non-singular maps, it has degree 1, so the covering has one sheet. Therefore each subset layer is a global diffeomorphism.

Full transform:
$$
F = F_7 \circ \cdots \circ F_0
$$

Composition of globally invertible subset layers gives a globally invertible field transformation.

---

## 11. Final Conclusions

### U(1)

- Jacobian scalar
- strictly positive
- globally invertible

### U(2)

Under invertibility assumptions:

- CNN depends only on frozen subsets
- current `base` coefficient caps hold

We have:

$$
\boxed{
\text{The full U(2) base field transformation is globally invertible}
}
$$

This is a mathematical exact-arithmetic statement. Numerically, the current caps have little margin near saturated `tanh`, and the implemented inverse still relies on fixed-point iteration convergence.

This statement is about invertibility, not gauge covariance. The current U(2)
base implementation should not be claimed to preserve gauge symmetry unless
its CNN inputs are restricted to gauge-invariant scalar features and its
traceless updates are built from gauge-covariant algebra bases at the active
link start point.

---

## 12. Summary

| Property | U(1) | U(2) |
|---|---|---|
| Jacobian type | scalar | $4\times4$ block |
| Positive definite | trivial | not applicable |
| Non-singular | guaranteed | guaranteed by current `base` caps |
| Determinant sign | positive | positive |
| Invertibility | global | global for current `base` caps |
| Gauge covariance | automatic for scalar loop features | requires scalar CNN features and covariant adjoint bases |
