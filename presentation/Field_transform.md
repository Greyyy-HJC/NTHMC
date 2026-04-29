# Detailed Comparison: 2D U(1) vs 2D U(2) Field Transformations

## 1. Core Principle: Gauge Covariance

Gauge transformation:
$$
U_{x,\mu} \rightarrow G_x U_{x,\mu} G_{x+\hat\mu}^\dagger
$$

A valid field transformation \(F\) must **commute with gauge transformation**:
$$
F(GUG^\dagger) = G F(U) G^\dagger
$$

### Consequence

The transformation must be built from **gauge-covariant objects**, i.e.:

- Wilson loops (plaquette, rectangle, etc.)
- Local gauge-covariant combinations

This is the fundamental constraint on allowed terms.

---

## 2. 2D U(1) Field Transformation

### Link variable

$$
U_{x,\mu} = e^{i\theta_{x,\mu}}
$$

### Transformation

$$
U'_{x,\mu} = e^{i\Delta\theta_{x,\mu}} U_{x,\mu}
$$

with

$$
\Delta\theta_{x,\mu}
=
\sum_p \left[
\epsilon_p^{(s)} \sin(\theta_p)
+
\epsilon_p^{(c)} \cos(\theta_p)
\right]
+
\sum_r \left[
\epsilon_r^{(s)} \sin(\theta_r)
+
\epsilon_r^{(c)} \cos(\theta_r)
\right]
$$

### Interpretation

- \(\sin\theta\): Lie algebra projection of loop
- \(\cos\theta\): additional basis allowed by symmetry

### Key properties

- Abelian → commuting
- Scalar degree of freedom
- Transform is additive

---

## 3. U(1) Jacobian

Because everything commutes:

$$
\theta' = \theta + \Delta\theta(\theta)
$$

$$
J = 1 + \frac{\partial \Delta\theta}{\partial \theta}
$$

Example:

$$
J
=
1
+
\sum_p \left[
\epsilon_p^{(s)} \cos(\theta_p)
-
\epsilon_p^{(c)} \sin(\theta_p)
\right]
+
\sum_r \left[
\epsilon_r^{(s)} \cos(\theta_r)
-
\epsilon_r^{(c)} \sin(\theta_r)
\right]
$$

So:

$$
\log \det J
=
\log\!\left(
1 + \frac{\partial \Delta\theta}{\partial \theta}
\right)
$$

This is why U(1) is simple.

---

## 4. 2D U(2) Field Transformation

### Link variable

$$
U_{x,\mu} \in U(2)
$$

### Transformation

$$
U'_{x,\mu} = e^{i\Delta_{x,\mu}(U)} U_{x,\mu}
$$

where

$$
\Delta_{x,\mu} \in u(2)
$$

$$
\Delta = a_0 I + \sum_{a=1}^3 a_a \sigma^a
$$

---

## 5. Allowed Terms in U(2)

Given a Wilson loop \(W_l \in U(2)\):

### Sin-like (Lie algebra projection)

$$
S(W) = \frac{W - W^\dagger}{2i}
$$

### Cos-like (Hermitian part)

$$
C(W) = \frac{W + W^\dagger}{2}
$$

These generalize:

$$
\sin\theta \rightarrow S(W), \quad \cos\theta \rightarrow C(W)
$$

---

## 6. Trace vs Traceless Decomposition

Each matrix splits into:

$$
S = S^{\text{trace}} + S^{\text{traceless}}
$$
$$
C = C^{\text{trace}} + C^{\text{traceless}}
$$

So full basis per loop:

- sin-like trace
- sin-like traceless
- cos-like trace
- cos-like traceless

---

## 7. Full U(2) Transform Structure

$$
\Delta_{x,\mu}
=
\sum_l \Big[
\epsilon_l^{(S,\text{tr})} S_l^{\text{tr}}
+
\epsilon_l^{(S,\text{tl})} S_l^{\text{tl}}
+
\epsilon_l^{(C,\text{tr})} C_l^{\text{tr}}
+
\epsilon_l^{(C,\text{tl})} C_l^{\text{tl}}
\Big]
$$

with loops:

- plaquette
- rectangle

---

## 8. Key Difference: Non-Abelian Nature

| Feature | U(1) | U(2) |
|------|------|------|
| Algebra dim | 1 | 4 |
| Commutativity | Yes | No |
| Transform | additive | exponential |
| Basis | sin, cos | matrix projections |
| Jacobian | scalar | matrix |

---

## 9. U(2) Jacobian

Define local perturbation:

$$
U_X = e^{iX} U
$$

Apply transform:

$$
U'_X = F(U_X)
$$

Extract output tangent:

$$
Y = \log(U'_X U'^\dagger)
$$

Then:

$$
J_{AB} = \frac{\partial Y^A}{\partial X^B}
$$

Each link → 4×4 Jacobian block.

In the implementation, the manual U(2) Jacobian propagates active-link tangent vectors through the plaquette and rectangle loops, then accumulates the log determinant of these local blocks. Small diagnostic runs can compare this path against an autograd Jacobian.

Total:

$$
\log \det J = \sum_{\text{links}} \log \det J_{\text{local}}
$$

---

## 10. Why U(2) is Harder

Because:

$$
e^{A+B} \neq e^A e^B
$$

and:

$$
\delta e^{i\Delta} \neq i\,\delta\Delta\, e^{i\Delta}
$$

Non-commutativity introduces:

- adjoint rotation
- BCH corrections
- non-trivial tangent map

---

## 11. Subset Trick

Split lattice into 8 subsets:

$$
(x \bmod 2, y \bmod 2, \mu)
$$

Ensures:

$$
\frac{\partial \Delta(x)}{\partial U(y)} = 0 \quad (x \neq y)
$$

→ Jacobian becomes block diagonal.

---

## 12. Final Comparison

### U(1):

$$
\theta \rightarrow \theta + \Delta\theta
$$

$$
\log \det J = \log(1 + \partial \Delta\theta)
$$

---

### U(2):

$$
U \rightarrow e^{i\Delta(U)} U
$$

$$
\log \det J = \sum \log \det \left(\frac{\partial Y}{\partial X}\right)
$$

---

## 13. Key Insight

U(2) is not just adding more terms.

It fundamentally changes:

- scalar → Lie algebra vector
- commuting → non-commuting
- derivative → tangent map

This is the conceptual jump from U(1) to non-Abelian gauge theory.
