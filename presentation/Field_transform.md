# Details of Field Transformation: 2D U(1) vs 2D U(2)

This note documents the field transformations implemented in:

- `src/nthmc/u1/field_transform.py`
- `src/nthmc/u2/field_transform.py`
- `src/nthmc/u1/models.py`
- `src/nthmc/u2/models.py`

---

## Representation of U(2)

$$
U = (\phi, q), \quad q \in SU(2)
$$

This is the split representation used by the current `src/nthmc/u2` code.
Each link is stored as a real tensor with last dimension 5:
$$
U_{x,\mu}
\leftrightarrow
(\phi_{x,\mu}, q_{0,x,\mu}, q_{1,x,\mu}, q_{2,x,\mu}, q_{3,x,\mu}),
\qquad
q_0^2+q_1^2+q_2^2+q_3^2=1.
$$

The corresponding complex matrix is
$$
U
=
e^{i\phi}
\begin{pmatrix}
q_0+i q_3 & q_2+i q_1 \\
-q_2+i q_1 & q_0-i q_3
\end{pmatrix}.
$$

Thus the U(2) degree of freedom is handled as a central U(1) phase times an
SU(2) unit quaternion. In group operations, the code keeps these two parts
separate:

- `u2_normalize` wraps $\phi$ to $[-\pi,\pi)$ and normalizes $q$.
- `u2_mul` adds the U(1) phases and multiplies the SU(2) quaternions.
- `u2_conj` negates the phase and quaternion-conjugates the SU(2) part.
- `u2_exp` maps four real algebra coefficients to
  $(\phi,\exp_{SU(2)} a)$, with one central phase direction and three
  traceless SU(2) directions.

Equivalently, the tangent algebra used by HMC and the field transform is
represented as
$$
u(2) \simeq u(1)\oplus su(2),
\qquad
\Delta = (\Delta_\phi,\Delta_1,\Delta_2,\Delta_3).
$$

The matrix-to-split conversion follows the same convention:
$$
\phi = \frac{1}{2}\arg\det U,
\qquad
q = e^{-i\phi}U \in SU(2),
$$
then the SU(2) matrix is converted to the quaternion components above.
Because $U(2)\simeq (U(1)\times SU(2))/\mathbb{Z}_2$, this split is a
coordinate convention with the usual sign/phase identification; the code fixes
a representative by wrapping the phase and normalizing the quaternion.


---

## Wilson Actions Implemented In Code

The current U(1) HMC code stores each link as one compact angle
$\theta_{x,\mu}$. The plaquette angle implemented by
`src/nthmc/u1/u1_observables.py::plaq_from_field` is
$$
\theta_p(x)
=
\theta_{x,0}
-
\theta_{x,1}
-
\theta_{x+\hat 1,0}
+
\theta_{x+\hat 0,1}.
$$

The action used by `src/nthmc/u1/u1_hmc.py::HMCU1.action` is
$$
S_{U(1)}(\theta)
=
-\beta\sum_x \cos\theta_p(x).
$$

This is equivalent for HMC dynamics to the usual Wilson form
$$
S_{U(1)}(\theta)
=
\beta\sum_x\left[1-\cos\theta_p(x)\right],
$$
because the two differ only by the constant $\beta V$.

The current U(2) code stores each link in the split representation
$$
U_{x,\mu}=e^{i\phi_{x,\mu}}q_{x,\mu},
\qquad
q_{x,\mu}\in SU(2),
$$
where $q=(q_0,q_1,q_2,q_3)$ is a unit quaternion. The plaquette matrix
implemented by `src/nthmc/u2/u2_observables.py::plaquette_from_field_batch`
has the same lattice orientation as the U(1) plaquette:
$$
P_{x,01}
=
U_{x,0}
U_{x,1}^\dagger
U_{x+\hat 1,0}^\dagger
U_{x+\hat 0,1}.
$$

In split form, the plaquette is stored as
$$
P_{x,01}=e^{i\phi_p(x)}q_p(x),
\qquad
q_p=(q_{0,p},q_{1,p},q_{2,p},q_{3,p}).
$$

The normalized plaquette used by
`src/nthmc/u2/u2_observables.py::plaquette_mean_from_field_batch` is
$$
\frac{1}{2}\mathrm{ReTr}\,P_{x,01}
=
q_{0,p}(x)\cos\phi_p(x).
$$

This is exactly the code expression
```python
torch.cos(plaquettes[..., 0]) * plaquettes[..., 1]
```
because `plaquettes[..., 0]` is the U(1) phase $\phi_p$ and
`plaquettes[..., 1]` is the scalar quaternion component $q_{0,p}$.

Therefore `src/nthmc/u2/u2_observables.py::action_from_field_batch` computes
$$
S_{U(2)}(U)
=
\beta V
\left[
1
-
\frac{1}{V}\sum_x
q_{0,p}(x)\cos\phi_p(x)
\right],
$$
or equivalently
$$
S_{U(2)}(U)
=
\beta\sum_x
\left[
1-\frac{1}{2}\mathrm{ReTr}\,P_{x,01}
\right].
$$

Thus the extra factor `plaquettes[..., 1]` is not an additional ad hoc
interaction. It is the SU(2) trace contribution required by the U(2) Wilson
action in the split $U(1)\times SU(2)$ coordinate convention.


---

## Topological charge definitions

The current 2du1 implementation uses the compact U(1) plaquette angle
$$
\theta_p(x)
=
\mathrm{wrap}_{[-\pi,\pi)}
\left[
\theta_{x,0}
-
\theta_{x,1}
-
\theta_{x+\hat 1,0}
+
\theta_{x+\hat 0,1}
\right],
$$
and records the integer-valued topological charge
$$
Q_{\mathrm{2du1}}
=
\left\lfloor
0.1
+
\frac{1}{2\pi}\sum_x \theta_p(x)
\right\rfloor .
$$

The current 2du2 implementation forms the U(2) plaquette matrix in the same
lattice orientation,
$$
P_{x,01}
=
U_{x,0}
U_{x,1}^\dagger
U_{x+\hat 1,0}^\dagger
U_{x+\hat 0,1},
$$
then uses the determinant phase:
$$
\alpha_p(x)
=
\mathrm{wrap}_{[-\pi,\pi)}
\left[
\arg\det P_{x,01}
\right].
$$

In the split U(2) representation used by the code, $U=e^{i\phi}q$ with
$q\in SU(2)$, so $\arg\det P_{x,01}=2\phi_p(x)$. Therefore the recorded
topological charge is
$$
Q_{\mathrm{2du2}}
=
\left\lfloor
0.1
+
\frac{1}{2\pi}\sum_x \alpha_p(x)
\right\rfloor .
$$

The small $0.1$ offset is part of the current implementation's integer
rounding convention.


---

## Gauge Covariance of Field Transformation

Gauge transformation:
$$
U_{x,\mu} \rightarrow U^G_{x,\mu} \equiv G_{x+\hat\mu} U_{x,\mu} G_x^\dagger .
$$

A gauge-covariant field transformation $F$ should satisfy:
$$
F(U^G) = F(U)^G .
$$

For Abelian groups like U(1), this requirement is straightforward to preserve
with ordinary scalar loop features. A closed U(1) Wilson loop is a phase
$C_x=e^{i\theta_x}$, and because all group elements commute, the gauge
factors cancel around the loop. Therefore the loop angle $\theta_x$, and
features such as $\sin\theta_x$ and $\cos\theta_x$, are gauge-invariant
scalars. Feeding these scalar channels to an ordinary CNN does not introduce
a local gauge-frame dependence.

For non-Abelian groups, this statement is not enough by itself. A closed
loop matrix based at site $x$ transforms by conjugation:
$$
C_x \rightarrow G_x C_x G_x^\dagger ,
$$
so its trace-like scalar features are gauge invariant, but its traceless
color components are only gauge covariant in the color frame at the loop base
point. In the split U(2) representation, this is the distinction between
trace-like quantities such as $q_0\cos\phi$ and $q_0\sin\phi$, which are
scalars, and traceless color-vector quantities such as $(q_1,q_2,q_3)$, which
rotate under the local adjoint action of $G_x$.

Therefore, gauge-invariant inputs in U(2) for the CNN should be built from
closed-loop matrix invariants. For a closed loop
$$
C=e^{i\phi}q,
\qquad
q=q_0I+i\sum_{a=1}^3q_a\sigma_a\in SU(2),
$$
use scalar quantities such as
$$
\mathrm{ReTr}\,C=2q_0\cos\phi,
\qquad
\mathrm{ImTr}\,C=2q_0\sin\phi.
$$

The determinant is also gauge invariant:
$$
\det C=e^{2i\phi},
\qquad
\mathrm{Re}\det C=\cos(2\phi),
\qquad
\mathrm{Im}\det C=\sin(2\phi).
$$

More generally,
$$
\mathrm{Tr}\,C^n
=
2e^{in\phi}T_n(q_0),
$$
where $T_n$ is the Chebyshev polynomial defined by
$T_n(\cos\alpha)=\cos(n\alpha)$. Equivalently,
$$
\mathrm{ReTr}\,C^n
=
2T_n(q_0)\cos(n\phi),
\qquad
\mathrm{ImTr}\,C^n
=
2T_n(q_0)\sin(n\phi).
$$
For example,
$$
\mathrm{Tr}\,C^2=2e^{2i\phi}(2q_0^2-1),
\qquad
\mathrm{Tr}\,C^3=2e^{3i\phi}(4q_0^3-3q_0).
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

## U(1) Field Transformation

### Update rule

The U(1) field transform uses the same 8 checkerboard subsets as the HMC
masking code:
$$
s=(\mu,\ x\bmod 2,\ y\bmod 2),
\qquad
\mu\in\{0,1\}.
$$

One subset is updated at a time. For subset $s$, only the active links selected
by `get_field_mask(s, ...)` are changed:
$$
\theta^{(s+1)}_{x,\mu}
=
\theta^{(s)}_{x,\mu}
+
M^{(s)}_{x,\mu}\Delta\theta^{(s)}_{x,\mu}(\theta^{(s)}),
$$
where $M^{(s)}_{x,\mu}\in\{0,1\}$ is the active-link mask. The full transform
is the composition of the 8 subset maps:
$$
F = F_7\circ F_6\circ\cdots\circ F_0.
$$

Within one subset, active links are independent because the CNN coefficients
are computed from the masked loop features before the active links are
updated.

### Input & Output of CNN

For each subset, the current U(1) code first computes the plaquette angles
$p_x$ and the two rectangle-angle fields $r^{(0)}_x,r^{(1)}_x$:
$$
\mathrm{plaq}=\texttt{plaq\_from\_field\_batch}(\theta),
\qquad
\mathrm{rect}=\texttt{rect\_from\_field\_batch}(\theta).
$$

The model input is built in `compute_k0_k1` from masked loop features:
$$
\texttt{plaq\_features}
=
(\sin p,\cos p),
\qquad
\texttt{rect\_features}
=
(\sin r^{(0)},\sin r^{(1)},\cos r^{(0)},\cos r^{(1)}).
$$

Thus the plaquette CNN input has 2 channels and the rectangle CNN input has 4
channels. The model returns
$$
k^{(p)}\in\mathbb{R}^{8},
\qquad
k^{(r)}\in\mathbb{R}^{16}
$$
per lattice site. The channel layout is:

- plaquette: 4 sin coefficients followed by 4 cos coefficients.
- rectangle: 8 sin coefficients followed by 8 cos coefficients.

In the current `addcos` model, these are bounded as
$$
|k^{(p)}|<\frac{1}{5},
\qquad
|k^{(r)}|<\frac{1}{40}.
$$

The older `base` model has the same output layout, but sets the cos
coefficient channels to zero.

### Transformation

For one active link $\ell=(x,\mu)$, the code stacks the loop angles touching
that link:

- two plaquette angles for each link direction, selected from the 4-channel
  `plaq_angles` stack.
- four rectangle angles for each link direction, selected from the 8-channel
  `rect_angles` stack.

Let these relevant loop angles be $a_l$, with orientation signs
$\sigma_l=\pm1$ matching `_plaq_phase_shift` and `_rect_phase_shift`. The
code uses
$$
\sigma^{(p)}=(-1,+1,+1,-1),
\qquad
\sigma^{(r)}=(-1,+1,-1,+1,+1,-1,+1,-1).
$$
For each active link direction, the update uses the relevant two plaquette
entries and four rectangle entries from these stacks. The
implemented phase update has the form
$$
\Delta\theta_\ell
=
\sum_{l\in P(\ell)}
\sigma_l
\left[
k^{(p,\sin)}_l\sin a_l
-
k^{(p,\cos)}_l\cos a_l
\right]
+
\sum_{l\in R(\ell)}
\sigma_l
\left[
k^{(r,\sin)}_l\sin a_l
-
k^{(r,\cos)}_l\cos a_l
\right].
$$

This is the formula implemented by `_plaq_phase_shift` and
`_rect_phase_shift`. The sin coefficient channels multiply signed
$\sin(\text{loop angle})$ terms, and the cos coefficient channels multiply the
oppositely signed $\cos(\text{loop angle})$ terms.

For the local Jacobian, the coefficients are treated as independent of the
active link within the current subset. Since the loop-angle orientation in the
code gives
$$
\frac{\partial a_l}{\partial\theta_\ell}=-\sigma_l,
$$
the active-link scalar Jacobian is
$$
J_\ell
=
\frac{\partial\theta'_\ell}{\partial\theta_\ell}
=
1
-
\sum_{l\in P(\ell)}
\left[
k^{(p,\sin)}_l\cos a_l
+
k^{(p,\cos)}_l\sin a_l
\right]
-
\sum_{l\in R(\ell)}
\left[
k^{(r,\sin)}_l\cos a_l
+
k^{(r,\cos)}_l\sin a_l
\right].
$$

This matches `_plaq_jac_shift` and `_rect_jac_shift`, where the Jacobian shift
is built from the `[-cos(loop), -sin(loop)]` stack. In the older `base` model,
the cos coefficient channels are zero, so the Jacobian shift reduces to the
terms proportional to $-\ k_l\cos a_l$.

The subset log determinant is therefore
$$
\log|\det J_s|
=
\sum_{\ell\in s}\log J_\ell,
$$
which is the `torch.log(1 + plaq_jac_shift + rect_jac_shift)` expression in
`compute_jac_logdet`.


---

## 4. U(2) Field Transformation

### Representation


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

For a U(2) loop matrix $C$, the code uses the same split representation:
$$
C=e^{i\phi}q,
\qquad
q=q_0I+i\sum_{a=1}^3 q_a\sigma_a .
$$

Expanding the complex matrix gives
$$
C
=
q_0\cos\phi\, I
+
q_0\sin\phi\, iI
+
\sum_a q_a\cos\phi\, i\sigma_a
-
\sum_a q_a\sin\phi\, \sigma_a .
$$

The function `loop_sin_cos_features(C)` stores these components as 8 real
channels:
$$
\mathrm{sin\_like}(C)
=
\left(
q_0\sin\phi,\ q_1\cos\phi,\ q_2\cos\phi,\ q_3\cos\phi
\right),
$$
$$
\mathrm{cos\_like}(C)
=
\left(
q_0\cos\phi,\ -q_1\sin\phi,\ -q_2\sin\phi,\ -q_3\sin\phi
\right).
$$

So the feature layout is:

- channel 0: central $iI$ sin-like scalar, $q_0\sin\phi$
- channels 1:4: traceless $i\sigma_a$ sin-like color vector,
  $q_a\cos\phi$
- channel 4: central $I$ cos-like scalar, $q_0\cos\phi$
- channels 5:8: traceless $\sigma_a$ cos-like color vector,
  $-q_a\sin\phi$

The old base implementation had two distinct objects:

1. **CNN input features.** For each loop $l$, the input was the feature vector
   $$
   f_l(C_l)
   =
   \left(
   q_0\sin\phi,\ q_1\cos\phi,\ q_2\cos\phi,\ q_3\cos\phi,\ 
   q_0\cos\phi,\ -q_1\sin\phi,\ -q_2\sin\phi,\ -q_3\sin\phi
   \right)_l .
   $$
   These are the values returned by `loop_sin_cos_features`. In the old
   implementation, all eight components were treated as ordinary scalar CNN
   input channels.

2. **CNN output coefficients.** The CNN output did not directly output
   $\Delta$. It output local coefficients for the field transform. In the full
   field-transform layout, each loop $l$ had four coefficient slots:
   $$
   k_l=(k_{l,0},k_{l,1},k_{l,2},k_{l,3}).
   $$
   For the four plaquette loops this gives 16 output channels; for the eight
   rectangle loops this gives 32 output channels.

The `_loop_delta` code then combined the CNN input features $f_l$ and CNN
output coefficients $k_l$ to build the algebra update. For one loop $l$, with
orientation sign $s_l=\pm1$, the contribution had the structure
$$
\Delta_{\phi,l}
=
k_{l,0}\,s_l\,q_0\sin\phi
+
k_{l,2}\,q_0\cos\phi,
$$
$$
\Delta_{a,l}
=
k_{l,1}\,s_l\,q_a\cos\phi
-
k_{l,3}\,q_a\sin\phi,
\qquad a=1,2,3.
$$

Here $k_{l,r}$ is a CNN output coefficient, while the $q_0\sin\phi$,
$q_a\cos\phi$, $q_0\cos\phi$, and $-q_a\sin\phi$ factors are loop features
computed from the gauge field.

Thus the old transform used both central U(1)-like loop scalars and
traceless SU(2)-like color-vector loop structures to build
$$
\Delta_l
=
(\Delta_{\phi,l},\Delta_{1,l},\Delta_{2,l},\Delta_{3,l})
\in u(1)\oplus su(2).
$$

This is sufficient for the local Jacobian analysis below, but it does **not**
guarantee U(2) gauge covariance. The central scalar pieces
$q_0\sin\phi$ and $q_0\cos\phi$ are trace-like gauge-invariant loop scalars.
The vectors $(q_1,q_2,q_3)$ are different: for a closed loop based at a site,
they transform by a local adjoint SO(3) color rotation under gauge
transformations. The old CNN treated those vector components as fixed scalar
channels, so its coefficients could depend on the arbitrary local color frame.
Therefore the old base transform could learn gauge-frame-dependent updates.

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

The CNN outputs only sin-like phase coefficients, matching the current U(1)
`base` model:

- plaquette: 4 loops times 1 sin-like phase slot = 4 nonzero channels
- rectangle: 8 loops times 1 sin-like phase slot = 8 nonzero channels

These are expanded back to the full field-transform layout:

- plaquette full layout: 16 channels
- rectangle full layout: 32 channels

For each loop, coefficient slot 0 is the sin-like phase slot and coefficient
slot 2 is the cos-like phase slot, while slots 1 and 3 are traceless color
slots. The current base model sets slots 1, 2, and 3 to zero. Thus the total
full-layout output has 12 nonzero sin-like phase channels and 36 identically
zero channels.

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



---

## 5. U(2) Jacobian Structure

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

## 6. Decomposition of the Jacobian

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

## 7. Invertibility Condition

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

## 8. Current U(2) Base Caps

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

## 9. Determinant Sign

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

## 10. Gauge-Covariant U(2) Design

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

## 11. Global Invertibility

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

## 12. Final Conclusions

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

## 13. Summary

| Property | U(1) | U(2) |
|---|---|---|
| Jacobian type | scalar | $4\times4$ block |
| Positive definite | trivial | not applicable |
| Non-singular | guaranteed | guaranteed by current `base` caps |
| Determinant sign | positive | positive |
| Invertibility | global | global for current `base` caps |
| Gauge covariance | automatic for scalar loop features | requires scalar CNN features and covariant adjoint bases |
