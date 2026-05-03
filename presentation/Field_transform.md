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

Gauge transformation:
$$
U_{x,\mu} \rightarrow G_x U_{x,\mu} G_{x+\hat\mu}^\dagger .
$$

A field transformation $F$ satisfies:
$$
F(GUG^\dagger) = G F(U) G^\dagger .
$$

Transformations are built from local Wilson loops:

- plaquettes
- $1\times2$, $2\times1$ rectangles

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

### Key structural property

**CNN coefficients depend only on frozen subsets**, not on active links:
$$
\frac{\partial k}{\partial U_{\text{active}}} = 0
$$

Thus Jacobian derivatives only arise from loop features.

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

These caps apply to all 16 plaquette coefficient channels and all 32 rectangle coefficient channels.

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

## 9. Global Invertibility

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

## 10. Final Conclusions

### U(1)

- Jacobian scalar
- strictly positive
- globally invertible

### U(2)

Under assumptions:

- CNN depends only on frozen subsets
- current `base` coefficient caps hold

We have:

$$
\boxed{
\text{The full U(2) base field transformation is globally invertible}
}
$$

This is a mathematical exact-arithmetic statement. Numerically, the current caps have little margin near saturated `tanh`, and the implemented inverse still relies on fixed-point iteration convergence.

---

## 11. Summary

| Property | U(1) | U(2) |
|---|---|---|
| Jacobian type | scalar | $4\times4$ block |
| Positive definite | trivial | not applicable |
| Non-singular | guaranteed | guaranteed by current `base` caps |
| Determinant sign | positive | positive |
| Invertibility | global | global for current `base` caps |
