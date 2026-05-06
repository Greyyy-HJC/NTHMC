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
U_{x+\hat 0,1}
U_{x+\hat 1,0}^\dagger
U_{x,1}^\dagger.
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

### U(1) and U(2) beta normalization

The numerical value of $\beta$ in the U(1) action and the U(2) action should
not be compared as the same effective coupling. The U(1) plaquette angle
appears as
$$
S_{U(1)}
=
\beta_{U(1)}\sum_x\left[1-\cos\theta_p(x)\right].
$$
For small plaquette angles,
$$
1-\cos\theta_p
=
\frac{1}{2}\theta_p^2+O(\theta_p^4),
$$
so the quadratic U(1) action is
$$
S_{U(1)}
\simeq
\frac{\beta_{U(1)}}{2}\sum_x\theta_p(x)^2.
$$

For U(2), write the plaquette as
$$
P_x=e^{i\phi_p(x)}q_p(x),
\qquad
q_p=\exp_{SU(2)}(i a_p^b\sigma_b).
$$
The determinant phase used by the U(2) topological charge is
$$
\alpha_p(x)=\arg\det P_x=2\phi_p(x),
$$
because $\det q_p=1$. For topological-freezing comparisons, $\alpha_p$ is the
U(2) variable to align with the U(1) plaquette angle $\theta_p$, because both
topological charges are built from the wrapped compact phase summed over
plaquettes. Expanding the normalized trace near the identity gives
$$
1-\frac{1}{2}\mathrm{ReTr}\,P_x
=
1-q_{0,p}\cos\phi_p
\simeq
\frac{1}{2}\phi_p^2
+
\frac{1}{2}|a_p|^2.
$$
The part controlling the determinant U(1) phase is therefore
$$
\frac{\beta_{U(2)}}{2}\phi_p^2
=
\frac{\beta_{U(2)}}{8}\alpha_p^2.
$$
Matching this quadratic coefficient to the U(1) action
$\frac{\beta_{U(1)}}{2}\theta_p^2$ with $\theta_p$ identified with the
determinant phase $\alpha_p$ gives the rough determinant-sector relation
$$
\frac{\beta_{U(2)}}{8}
\approx
\frac{\beta_{U(1)}}{2},
\qquad\text{or}\qquad
\beta_{U(2)}
\approx
4\beta_{U(1)}.
$$

This is only a normalization guide, not an exact equality of the two theories.
The U(2) theory also has the three traceless SU(2) plaquette directions, and
topological freezing depends on the full dynamics and the chosen HMC
integrator. Still, for determinant-topology comparisons, a U(2) run at
$\beta_{U(2)}$ is more naturally compared to a U(1) run at the effective value
$$
\beta_{\mathrm{det,eff}}
\equiv
\frac{\beta_{U(2)}}{4}.
$$


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
U_{x+\hat 0,1}
U_{x+\hat 1,0}^\dagger
U_{x,1}^\dagger,
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
U_{x,\mu} \rightarrow U^G_{x,\mu} \equiv G_x U_{x,\mu} G_{x+\hat\mu}^\dagger .
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

For U(1), each link is represented by a compact angle
$$
U_{x,\mu}=e^{i\theta_{x,\mu}}.
$$
The field transformation is a phase shift:
$$
\theta'_{x,\mu}
=
\theta_{x,\mu}
+
\Delta\theta_{x,\mu}(\theta),
\qquad
U'_{x,\mu}
=
e^{i\Delta\theta_{x,\mu}(\theta)}U_{x,\mu}.
$$

### Subset update

Links are partitioned into 8 disjoint subsets and updated sequentially. The
subset label is
$$
s=(\mu,\ x\bmod 2,\ y\bmod 2),
\qquad
\mu\in\{0,1\}.
$$

For subset $s$, only the active links selected by `get_field_mask(s, ...)`
are changed:
$$
\theta^{(s+1)}_\ell
=
\theta^{(s)}_\ell
+
M^{(s)}_\ell\Delta\theta^{(s)}_\ell(\theta^{(s)}),
\qquad
\ell=(x,\mu),
$$
where $M^{(s)}_\ell\in\{0,1\}$ is the active-link mask. The full transform is
the composition of the 8 subset maps:
$$
F = F_7\circ F_6\circ\cdots\circ F_0.
$$

Within one subset, active links are independent because the CNN inputs are
masked so they do not depend on the active subset. Therefore each subset
Jacobian factorizes into scalar active-link Jacobians.

### Input & Output of CNN

For each subset, first compute the plaquette angle field $p_x$ and the two
rectangle-angle fields $r^{(0)}_x,r^{(1)}_x$:
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

The model returns
$$
k^{(p)}\in\mathbb{R}^{8},
\qquad
k^{(r)}\in\mathbb{R}^{16}
$$
per lattice site. The CNN channel summary is:

- input plaquette channels: 2, given by $(\sin p,\cos p)$.
- input rectangle channels: 4, given by
  $(\sin r^{(0)},\sin r^{(1)},\cos r^{(0)},\cos r^{(1)})$.
- output plaquette channels: 8, with 4 sin coefficients followed by 4 cos
  coefficients. The 4 plaquette coefficient slots cover both link directions:
  2 stack entries for $\mu=0$ and 2 stack entries for $\mu=1$.
- output rectangle channels: 16, with 8 sin coefficients followed by 8 cos
  coefficients. The 8 rectangle coefficient slots also cover both link
  directions: 4 stack entries for $\mu=0$ and 4 stack entries for $\mu=1$.

### Transformation

For one active link $\ell=(x,\mu)$, let $l$ label one oriented Wilson loop
touching that link. The loop index runs over the attached $1\times1$
plaquettes and $1\times2$ rectangles:
$$
l\in P(\ell)\cup R(\ell),
\qquad
|P(\ell)|=2,
\qquad
|R(\ell)|=4.
$$

The loop angles are stored in shifted stacks:

- two plaquette angles for each link direction, selected from the 4-channel
  `plaq_angles` stack.
- four rectangle angles for each link direction, selected from the 8-channel
  `rect_angles` stack.

Let $a_l$ denote the selected loop angle:
$$
a_l(\theta)
=
\begin{cases}
\text{the selected entry of } \texttt{\_plaq\_angle\_stack}(\mathrm{plaq}),
& l\in P(\ell),\\
\text{the selected entry of } \texttt{\_rect\_angle\_stack}(\mathrm{rect}),
& l\in R(\ell).
\end{cases}
$$
For link direction $\mu=0$, the active update uses plaquette stack channels
$0,1$ and rectangle stack channels $0,1,2,3$. For link direction $\mu=1$, it
uses plaquette stack channels $2,3$ and rectangle stack channels $4,5,6,7$.

The orientation signs are
$$
\sigma^{(p)}=(-1,+1,+1,-1),
\qquad
\sigma^{(r)}=(-1,+1,-1,+1,+1,-1,+1,-1).
$$
The phase shift is
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

This formula includes both the sine channels and the optional cos channels.
When the cos channels are zero, it reduces to the sine-only transformation.

### Jacobian

For the local Jacobian, the coefficients are treated as independent of the
active link within the current subset, because the CNN inputs are masked before
the active links are updated. Since
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
\right],
$$
matching `_plaq_jac_shift` and `_rect_jac_shift`, where the Jacobian shift is
built from the `[-cos(loop), -sin(loop)]` stack.

For the current bounded coefficient layout with cos terms, set
$$
c_{\rm plaq}=\frac15,
\qquad
c_{\rm rect}=\frac1{40}.
$$
The strict `tanh` bounds give
$$
|k^{(p,\sin)}_l|,\ |k^{(p,\cos)}_l| < c_{\rm plaq},
\qquad
|k^{(r,\sin)}_l|,\ |k^{(r,\cos)}_l| < c_{\rm rect}.
$$
Since $|\sin a_l|\le1$ and $|\cos a_l|\le1$,
$$
\left|
k^{(p,\sin)}_l\cos a_l
+
k^{(p,\cos)}_l\sin a_l
\right|
<
2c_{\rm plaq},
$$
and similarly each rectangle contribution is strictly bounded by
$2c_{\rm rect}$. Each active link sees 2 plaquette loops and 4 rectangle loops,
so
$$
\left|J_\ell-1\right|
<
2(2c_{\rm plaq})+4(2c_{\rm rect})
=
4c_{\rm plaq}+8c_{\rm rect}
=
4\cdot\frac15+8\cdot\frac1{40}
=
1.
$$
Therefore
$$
\boxed{J_\ell>0}
$$
for every active U(1) link in exact arithmetic. This is the explicit reason
the current $1/5$ and $1/40$ coefficient boundaries make the scalar Jacobian
positive even with cos channels.

The subset log determinant is therefore
$$
\log\det J_s
=
\sum_{\ell\in s}\log J_\ell,
$$
which is the `torch.log(1 + plaq_jac_shift + rect_jac_shift)` expression in
`compute_jac_logdet`.


---

## U(2) Field Transformation

For U(2), each link is represented in the split convention
$$
U_{x,\mu}=e^{i\phi_{x,\mu}}q_{x,\mu},
\qquad
q_{x,\mu}=q_0I+i\sum_{a=1}^3q_a\sigma_a\in SU(2).
$$
The tangent update lives in
$$
u(2)\simeq u(1)\oplus su(2),
\qquad
\Delta_{x,\mu}
=
(\Delta_{\phi,x,\mu},\Delta_{1,x,\mu},
\Delta_{2,x,\mu},\Delta_{3,x,\mu}).
$$

The field transformation uses left multiplication:
$$
U'_{x,\mu}
=
\exp_{U(2)}(\Delta_{x,\mu}(U))\,U_{x,\mu}.
$$
In the split storage this means
$$
\phi'_{x,\mu}
=
\phi_{x,\mu}+\Delta_{\phi,x,\mu},
\qquad
q'_{x,\mu}
=
\exp_{SU(2)}
\!\left(\sum_{a=1}^3\Delta_{a,x,\mu}\,i\sigma_a\right)
q_{x,\mu},
$$
followed by phase wrapping and quaternion normalization in the implementation.
For this update to be gauge covariant under
$$
U_{x,\mu}\rightarrow U^G_{x,\mu}
=G_xU_{x,\mu}G_{x+\hat\mu}^\dagger,
$$
the algebra update must transform in the site-$x$ color frame:
$$
\Delta_{x,\mu}(U^G)
=
G_x \Delta_{x,\mu}(U) G_x^\dagger .
$$
Equivalently, suppose the input field has already been gauge transformed,
$$
\widetilde U'_{x,\mu}
=
G_x\widetilde U_{x,\mu}G_{x+\hat\mu}^\dagger,
$$
and the field transformation is applied by left multiplication:
$$
U_{x,\mu}
=
e^{\Delta_{x,\mu}}\widetilde U_{x,\mu},
\qquad
U'_{x,\mu}
=
e^{\Delta'_{x,\mu}}\widetilde U'_{x,\mu}.
$$
Gauge covariance requires $U'_{x,\mu}=G_xU_{x,\mu}G_{x+\hat\mu}^\dagger$.
Substituting the left-multiplied update gives
$$
G_x e^{\Delta_{x,\mu}}\widetilde U_{x,\mu}
G_{x+\hat\mu}^\dagger
=
e^{\Delta'_{x,\mu}}
G_x\widetilde U_{x,\mu}G_{x+\hat\mu}^\dagger .
$$
After canceling the common right factor, the required condition is
$$
G_x e^{\Delta_{x,\mu}}G_x^\dagger
=
e^{\Delta'_{x,\mu}},
\qquad\text{so}\qquad
\Delta'_{x,\mu}
=
G_x\Delta_{x,\mu}G_x^\dagger .
$$
This fixes the gauge frame of every non-Abelian object used to build
$\Delta_{x,\mu}$. If a loop contribution uses the traceless components of a
closed loop $C_l$, then that loop must transform in the same site-$x$ frame:
$$
C_l(U^G)=G_x C_l(U)G_x^\dagger .
$$
Therefore each attached plaquette or rectangle in the $\Delta_{x,\mu}$ loop
stack must be written as a cyclic Wilson loop that starts and ends at the
active link's starting site $x$. A closed loop based at another site would
rotate in a different local gauge frame, and its color-vector components could
not be added directly to the left-multiplied update for $U_{x,\mu}$.

### Subset update

The U(2) transform uses the same 8-subset link partition as U(1):
$$
s=(\mu,\ x\bmod 2,\ y\bmod 2),
\qquad
\mu\in\{0,1\}.
$$
For subset $s$, only the active links selected by the link mask are changed:
$$
U^{(s+1)}_\ell
=
\exp_{U(2)}
\!\left(M^{(s)}_\ell\Delta^{(s)}_\ell(U^{(s)})\right)
U^{(s)}_\ell,
\qquad
\ell=(x,\mu),
$$
where $M^{(s)}_\ell\in\{0,1\}$ is the active-link mask. The full
transformation is the composition
$$
F = F_7\circ F_6\circ\cdots\circ F_0.
$$
As in U(1), the loop inputs for one subset are masked so that the CNN
coefficients depend only on frozen links, not on the active links in that
subset:
$$
\frac{\partial k}{\partial U_{\mathrm{active}}}=0.
$$
Therefore the local Jacobian derivatives come from the explicit loop factors
inside the update, not from differentiating the CNN coefficients through the
active links.

### Input & Output of CNN

For each subset, first compute the closed U(2) plaquette loops and the two
closed rectangle-loop orientations:
$$
\mathrm{plaq}=\texttt{plaquette\_from\_field\_batch}(U),
\qquad
\mathrm{rect}=\texttt{rectangle\_from\_field\_batch}(U).
$$
For U(2), the rectangle multiplication order is part of the definition: the
product must be a closed non-Abelian Wilson loop. Unlike U(1), changing the
order changes the matrix-valued loop.

These plaquette and rectangle tensors are used only to build the
gauge-invariant scalar inputs for the CNN coefficients. The update
$\Delta_{x,\mu}$ is built from a separate attached loop stack,
`_plaq_loop_stack(U)` and `_rect_loop_stack(U)`, whose entries are cyclically
represented as Wilson loops based at the active link's site $x$. This
separation is harmless for the CNN inputs because their scalar traces and
determinants do not depend on the loop base point, but it is required for the
traceless components that enter $\Delta_{x,\mu}$.

For a closed loop
$$
C=e^{i\phi}q,
\qquad
q=q_0I+i\sum_{a=1}^3q_a\sigma_a\in SU(2),
$$
the ordinary CNN should receive only gauge-invariant scalar loop features.
The U(2) scalar input for one loop is
$$
\left(
q_0\cos\phi,
q_0\sin\phi,
\cos\phi,
\sin\phi,
2(2q_0^2-1)\cos(2\phi),
2(2q_0^2-1)\sin(2\phi)
\right).
$$
The first two entries are trace-like phase features, the next two encode the
central phase, and the final two are the real and imaginary parts of
$\mathrm{Tr}\,C^2$. With one plaquette orientation and two rectangle
orientations, this gives 6 plaquette input channels and 12 rectangle input
channels before the CNN combines them spatially.

The CNN output does not directly output $\Delta$. It outputs scalar
coefficient slots for the component-wise field transform. For each selected
loop $l$, the coefficient vector is
$$
k_l=(k_{l,0},k_{l,1},k_{l,2},k_{l,3}).
$$
The coefficients are scalar fields computed from the invariant CNN inputs
above. The CNN channel summary is:

- input plaquette channels: 6, one six-scalar feature vector for the plaquette
  loop.
- input rectangle channels: 12, from 2 rectangle orientations times 6 scalar
  features.
- output plaquette channels: 16, from 4 plaquette loops times 4 coefficient
  slots. The 4 plaquette loop slots cover both link directions: 2 for
  $\mu=0$ and 2 for $\mu=1$.
- output rectangle channels: 32, from 8 rectangle loops times 4 coefficient
  slots. The 8 rectangle loop slots cover both link directions: 4 for
  $\mu=0$ and 4 for $\mu=1$.

### Transformation

For one active link $\ell=(x,\mu)$, let $l$ label one oriented closed loop
touching that link:
$$
l\in P(\ell)\cup R(\ell),
\qquad
|P(\ell)|=2,
\qquad
|R(\ell)|=4.
$$
The 4 plaquette and 8 rectangle output-loop slots are the full stack layout
for both link directions. A fixed active link direction uses only the
corresponding half of those slots, which is why a single link has
$|P(\ell)|=2$ and $|R(\ell)|=4$.
For link direction $\mu=0$, the active update uses the two plaquette stack
entries associated with direction 0 and the four rectangle stack entries
associated with direction 0. For $\mu=1$, it uses the corresponding direction
1 stack entries. In the U(2) implementation each attached plaquette and
rectangle loop is cyclically represented as a closed Wilson loop based at the
active link's site-$x$ frame. The orientation signs are the same pattern as
U(1):
$$
\sigma^{(p)}=(-1,+1,+1,-1),
\qquad
\sigma^{(r)}=(-1,+1,-1,+1,+1,-1,+1,-1).
$$

For a selected loop
$$
C_l=e^{i\phi_l}q_l,
\qquad
q_l=q_{0,l}I+i\sum_{a=1}^3q_{a,l}\sigma_a,
$$
the split real features used by the field-transform formula are
$$
\mathrm{sin\_like}(C_l)
=
\left(
q_{0,l}\sin\phi_l,
q_{1,l}\cos\phi_l,
q_{2,l}\cos\phi_l,
q_{3,l}\cos\phi_l
\right),
$$
$$
\mathrm{cos\_like}(C_l)
=
\left(
q_{0,l}\cos\phi_l,
-q_{1,l}\sin\phi_l,
-q_{2,l}\sin\phi_l,
-q_{3,l}\sin\phi_l
\right).
$$
The four coefficient slots combine these real split components into one
$u(2)\simeq u(1)\oplus su(2)$ contribution:
$$
\Delta_{\phi,l}
=
k_{l,0}\,\sigma_l\,q_{0,l}\sin\phi_l
+
k_{l,2}\,q_{0,l}\cos\phi_l,
$$
$$
\Delta_{a,l}
=
k_{l,1}\,\sigma_l\,q_{a,l}\cos\phi_l
-
k_{l,3}\,q_{a,l}\sin\phi_l,
\qquad a=1,2,3.
$$
The active-link update is the sum over its attached plaquette and rectangle
loops:
$$
\Delta_\ell
=
\sum_{l\in P(\ell)\cup R(\ell)}
(\Delta_{\phi,l},\Delta_{1,l},\Delta_{2,l},\Delta_{3,l}).
$$
Because the attached loop stack is based at the active link's site-$x$ frame,
the loop component vector $(q_{1,l},q_{2,l},q_{3,l})$ and the resulting
$(\Delta_{1,l},\Delta_{2,l},\Delta_{3,l})$ transform in the correct local
adjoint frame.

### Jacobian

For the local Jacobian, perturb one active U(2) link by a tangent algebra
element $X\in u(2)$:
$$
U_X=\exp_{U(2)}(X)U.
$$
The output tangent is
$$
Y=\log_{U(2)}(U'_XU'^\dagger),
$$
so one active link contributes a real $4\times4$ Jacobian block
$$
J_\ell=\frac{\partial Y}{\partial X}.
$$

Within one subset, the CNN coefficients $k_l$ are independent of the active
links because of the input masks. Therefore the derivative is taken only
through the split loop variables $(\phi_l,q_l)$. If the active-link tangent
induces first-order loop variations
$$
\delta\phi_l,
\qquad
\delta q_{0,l},
\qquad
\delta q_{a,l},
$$
then the feature derivatives are
$$
\delta(q_{0,l}\sin\phi_l)
=
\delta q_{0,l}\sin\phi_l
+
q_{0,l}\cos\phi_l\,\delta\phi_l,
$$
$$
\delta(q_{a,l}\cos\phi_l)
=
\delta q_{a,l}\cos\phi_l
-
q_{a,l}\sin\phi_l\,\delta\phi_l,
$$
$$
\delta(q_{0,l}\cos\phi_l)
=
\delta q_{0,l}\cos\phi_l
-
q_{0,l}\sin\phi_l\,\delta\phi_l,
$$
$$
\delta(-q_{a,l}\sin\phi_l)
=
-\delta q_{a,l}\sin\phi_l
-
q_{a,l}\cos\phi_l\,\delta\phi_l.
$$
Thus one loop contributes
$$
\delta\Delta_{\phi,l}
=
k_{l,0}\sigma_l
\left[
\delta q_{0,l}\sin\phi_l
+
q_{0,l}\cos\phi_l\,\delta\phi_l
\right]
+
k_{l,2}
\left[
\delta q_{0,l}\cos\phi_l
-
q_{0,l}\sin\phi_l\,\delta\phi_l
\right],
$$
$$
\delta\Delta_{a,l}
=
k_{l,1}\sigma_l
\left[
\delta q_{a,l}\cos\phi_l
-
q_{a,l}\sin\phi_l\,\delta\phi_l
\right]
+
k_{l,3}
\left[
-\delta q_{a,l}\sin\phi_l
-
q_{a,l}\cos\phi_l\,\delta\phi_l
\right].
$$
These are the component-wise derivatives of the same four split features used
in the transformation formula.

The full active-link Jacobian block maps
$(X_\phi,X_1,X_2,X_3)$ to $(Y_\phi,Y_1,Y_2,Y_3)$. The subset Jacobian
factorizes over active links:
$$
\det J_s
=
\prod_{\ell\in s}\det J_\ell,
\qquad
\log|\det J_s|
=
\sum_{\ell\in s}\log|\det J_\ell|.
$$

The local block can be written as
$$
J_\ell = Q_\ell + E_\ell,
$$
where the two terms have different origins.

The $Q_\ell$ term is what remains if $\Delta_\ell$ is held fixed while the
input link is perturbed. Write
$$
\exp_{U(2)}(\Delta_\ell)
=
(\alpha_\ell,r_\ell),
\qquad
r_\ell=(r_0,r_1,r_2,r_3)\in SU(2),
$$
where $\alpha_\ell=\Delta_{\phi,\ell}$ is the central phase and
$r_\ell=\exp_{SU(2)}(\Delta_{1,\ell},\Delta_{2,\ell},\Delta_{3,\ell})$.
Because the central U(1) phase commutes with every algebra element,
the adjoint action leaves the $u(1)$ tangent component unchanged. The
traceless tangent vector is rotated by conjugation with the SU(2) quaternion:
$$
\mathrm{Ad}_{\exp(\Delta_\ell)}
\begin{pmatrix}
X_\phi\\
X_1\\
X_2\\
X_3
\end{pmatrix}
=
\begin{pmatrix}
X_\phi\\
\mathcal R(r_\ell)
\begin{pmatrix}
X_1\\
X_2\\
X_3
\end{pmatrix}
\end{pmatrix}.
$$
Thus, in the split basis,
$$
Q_\ell
=
\begin{pmatrix}
1 & 0\\
0 & \mathcal R(r_\ell)
\end{pmatrix}.
$$
For $v=(r_1,r_2,r_3)$ and
$$
[v]_\times
=
\begin{pmatrix}
0 & -r_3 & r_2\\
r_3 & 0 & -r_1\\
-r_2 & r_1 & 0
\end{pmatrix},
$$
the rotation matrix in the code's quaternion convention is
$$
\mathcal R(r)
=
(r_0^2-v^Tv)I_3
+
2vv^T
-
2r_0[v]_\times .
$$
Equivalently,
$$
\mathcal R(r)
=
\begin{pmatrix}
r_0^2+r_1^2-r_2^2-r_3^2
&
2(r_1r_2+r_0r_3)
&
2(r_1r_3-r_0r_2)
\\
2(r_1r_2-r_0r_3)
&
r_0^2-r_1^2+r_2^2-r_3^2
&
2(r_2r_3+r_0r_1)
\\
2(r_1r_3+r_0r_2)
&
2(r_2r_3-r_0r_1)
&
r_0^2-r_1^2-r_2^2+r_3^2
\end{pmatrix}.
$$
Since $r$ is a unit quaternion, conjugation preserves the Euclidean norm of a
pure quaternion:
$$
\left|r(0,X)r^\dagger\right|=|X|.
$$
Therefore $\mathcal R(r)^T\mathcal R(r)=I_3$ and
$\det\mathcal R(r)=1$. Hence
$$
\|Q_\ell\|_2=1,
\qquad
\det Q_\ell=1.
$$

The $E_\ell$ term is the remaining part caused by the active-link dependence
of the loop factors in $\Delta_\ell$. In differential notation,
$$
E_\ell X
=
D\exp_{\Delta_\ell}\!\left[D\Delta_\ell[X]\right],
$$
with the result expressed in the same output tangent coordinates as $Y$. This
is the term computed from the component derivatives of
$(\phi_l,q_{0,l},q_{1,l},q_{2,l},q_{3,l})$ above.

Concretely, the first step is the loop-feature derivative:
$$
D\Delta_\ell[X]
=
\delta\Delta_\ell
=
\sum_{l\in P(\ell)\cup R(\ell)}
(\delta\Delta_{\phi,l},
\delta\Delta_{1,l},
\delta\Delta_{2,l},
\delta\Delta_{3,l}).
$$
Each $\delta\Delta_{\phi,l}$ and $\delta\Delta_{a,l}$ is linear in $X$,
because the induced loop variations
$(\delta\phi_l,\delta q_{0,l},\delta q_{1,l},\delta q_{2,l},\delta q_{3,l})$
are linear in the active-link tangent. This is the quantity called
`delta_jac` in the code, built by `_plaq_delta_jac` and `_rect_delta_jac`.

The second step is the differential of the group exponential. Write
$$
\Delta_\ell=(\Delta_{\phi,\ell},\Delta_{\mathrm{vec},\ell}),
\qquad
\delta\Delta_\ell=(\delta\Delta_{\phi,\ell},
\delta\Delta_{\mathrm{vec},\ell}),
$$
where
$$
\Delta_{\mathrm{vec},\ell}
=
(\Delta_{1,\ell},\Delta_{2,\ell},\Delta_{3,\ell}).
$$
The central phase part is linear, so it passes through unchanged:
$$
(E_\ell X)_\phi=\delta\Delta_{\phi,\ell}.
$$
The SU(2) part is not simple addition; it is the left-trivialized
differential of the SU(2) exponential:
$$
(E_\ell X)_{\mathrm{vec}}
=
D\exp_{SU(2),\Delta_{\mathrm{vec},\ell}}
\left[\delta\Delta_{\mathrm{vec},\ell}\right].
$$
This is exactly the `_exp_tangent(delta, delta_jac)` contribution in the
manual Jacobian code.

Equivalently, the $4\times4$ matrix $E_\ell$ is obtained column by column:
for each tangent basis vector
$$
e_\phi,\ e_1,\ e_2,\ e_3,
$$
compute $D\Delta_\ell[e_j]$ from the attached loop-feature derivatives, pass
it through $D\exp_{\Delta_\ell}[\cdot]$, and place the resulting output
tangent vector as column $j$ of $E_\ell$.

Therefore
$$
J_\ell
=
Q_\ell(I+Q_\ell^{-1}E_\ell),
$$
and the sufficient non-singularity condition is
$$
\|E_\ell\|_2<1.
$$

With coefficient caps
$$
c_{\rm plaq}=\frac15,
\qquad
c_{\rm rect}=\frac1{40},
$$
one active link receives 2 plaquette-loop contributions and 4
rectangle-loop contributions. The conservative bound is
$$
\|E_\ell\|_2
<
4c_{\rm plaq}+8c_{\rm rect}
=
4\cdot\frac15+8\cdot\frac1{40}
=1,
$$
where the inequality is strict because the coefficients are strictly bounded
by their `tanh` caps for finite real network outputs. Hence each active-link
block is non-singular in exact arithmetic. The determinant sign is positive
by the homotopy $J_\ell(t)=Q_\ell+tE_\ell$, since the block cannot cross a
zero determinant along $t\in[0,1]$ and $\det J_\ell(0)=1$.

The full U(2) transform is the composition of the 8 subset maps. Since each
subset map has non-singular active-link blocks under the same bound, and the
scaled maps $\exp(t\Delta)$ connect each subset continuously to the identity,
the subset maps are globally invertible on the compact product of U(2) link
manifolds. Their composition is therefore globally invertible under these
coefficient bounds.

## Summary

The U(1) field transformation is a scalar compact phase shift. Its loop
inputs are ordinary gauge-invariant sine and cosine features, its active-link
Jacobian is a scalar, and the coefficient caps make that scalar strictly
positive. The 8-subset composition is therefore globally invertible.

The U(2) field transformation has the same subset structure but a larger
local tangent space. Each active link carries one central phase direction and
three traceless SU(2) directions, so the local Jacobian is a $4\times4$ real
block rather than a scalar. Gauge covariance requires the CNN to see only
gauge-invariant scalar loop inputs, such as $q_0\cos\phi$, $q_0\sin\phi$,
$\cos\phi$, $\sin\phi$, $2(2q_0^2-1)\cos(2\phi)$, and
$2(2q_0^2-1)\sin(2\phi)$. The update itself is written in the split
coordinates $(\phi,q_0,q_1,q_2,q_3)$. In the current implementation, every
traceless loop component used in the update is taken from a Wilson loop based
at the active link's starting site, so it transforms in the active link's
local color frame.

The same masking idea is essential in both groups: within one subset, the CNN
coefficients are computed from frozen links and do not depend on the active
links. For U(1), this gives scalar Jacobian factors. For U(2), this gives
independent $4\times4$ active-link blocks. Under the stated coefficient caps,
the U(2) perturbation part of each block has norm strictly below the
invertibility threshold, so each subset map is non-singular and the full
8-subset composition is globally invertible in exact arithmetic.
