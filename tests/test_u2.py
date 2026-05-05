import torch

from nthmc.u2.field_transform import FieldTransformation
from nthmc.u2.models import LocalNet
from nthmc.u2.u2_fthmc import HMCU2FT
from nthmc.u2.u2_hmc import HMCU2
from nthmc.u2.u2_observables import (
    action_from_field_batch,
    identity_field,
    loop_sin_cos_features,
    matrix_to_u2,
    plaq_mean_theory,
    plaquette_from_field_batch,
    plaquette_mean_from_field,
    plaquette_from_field,
    rectangle_from_field_batch,
    set_seed,
    topology_from_field,
    u2_conj,
    u2_exp,
    u2_log,
    u2_mul,
    u2_to_matrix,
)
from nthmc.u1.u1_observables import (
    plaq_from_field_batch as u1_plaq_from_field_batch,
    rect_from_field_batch as u1_rect_from_field_batch,
)


def gauge_transform_split(links: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Apply a local U(2) gauge transform to split batch links."""
    transformed = links.clone()
    transformed[:, 0] = u2_mul(u2_mul(torch.roll(omega, shifts=-1, dims=1), links[:, 0]), u2_conj(omega))
    transformed[:, 1] = u2_mul(u2_mul(torch.roll(omega, shifts=-1, dims=2), links[:, 1]), u2_conj(omega))
    return transformed


def test_u2_exp_returns_unitary_matrices() -> None:
    algebra = torch.randn(3, 2, 4, 4, 4)
    links = u2_exp(algebra)
    matrices = u2_to_matrix(links)
    identity = torch.eye(2, dtype=matrices.dtype)
    unitarity = matrices.mH @ matrices

    assert links.shape == (3, 2, 4, 4, 5)
    assert torch.allclose(unitarity, identity.expand_as(unitarity), atol=1e-5)


def test_u2_exp_has_unit_determinant_magnitude_with_free_phase() -> None:
    links = u2_exp(torch.randn(2, 4, 4, 4))
    determinant = torch.linalg.det(u2_to_matrix(links))

    assert torch.allclose(torch.abs(determinant), torch.ones_like(determinant.real), atol=1e-5)
    assert torch.any(torch.abs(torch.angle(determinant)) > 1e-5)


def test_split_u2_multiplication_matches_matrix_multiplication() -> None:
    left = u2_exp(torch.randn(2, 4, 4, 4))
    right = u2_exp(torch.randn(2, 4, 4, 4))

    split_product = u2_to_matrix(u2_mul(left, right))
    matrix_product = u2_to_matrix(left) @ u2_to_matrix(right)

    assert torch.allclose(split_product, matrix_product, atol=1e-5)


def test_matrix_to_u2_round_trip_matches_original_matrices() -> None:
    links = u2_exp(torch.randn(5, 2, 4, 4, 4))
    matrices = u2_to_matrix(links)
    recovered = matrix_to_u2(matrices)

    assert recovered.shape == links.shape
    assert torch.allclose(u2_to_matrix(recovered), matrices, atol=1e-5)


def test_batch_plaquette_matches_single_field_helper() -> None:
    links = u2_exp(torch.randn(3, 2, 4, 4, 4))
    batch_plaquettes = plaquette_from_field_batch(links)
    single_plaquettes = torch.stack([plaquette_from_field(link) for link in links])

    assert torch.allclose(u2_to_matrix(batch_plaquettes), u2_to_matrix(single_plaquettes), atol=1e-5)


def test_rectangle_from_field_batch_returns_unitary_loops() -> None:
    links = u2_exp(torch.randn(3, 2, 4, 4, 4))
    rectangles = rectangle_from_field_batch(links)
    matrices = u2_to_matrix(rectangles)
    identity = torch.eye(2, dtype=matrices.dtype)

    assert rectangles.shape == (3, 2, 4, 4, 5)
    assert torch.allclose(matrices.mH @ matrices, identity.expand_as(matrices), atol=1e-5)


def test_u2_loops_match_u1_angles_for_phase_embedded_fields() -> None:
    theta = torch.randn(2, 4, 4)
    links = torch.zeros(1, 2, 4, 4, 5)
    links[0, ..., 0] = theta
    links[..., 1] = 1.0

    u2_plaq_phase = plaquette_from_field_batch(links)[..., 0]
    u1_plaq = u1_plaq_from_field_batch(theta.unsqueeze(0))
    u2_rect_phase = rectangle_from_field_batch(links)[..., 0]
    u1_rect = u1_rect_from_field_batch(theta.unsqueeze(0))

    assert torch.allclose(torch.cos(u2_plaq_phase), torch.cos(u1_plaq), atol=1e-5)
    assert torch.allclose(torch.sin(u2_plaq_phase), torch.sin(u1_plaq), atol=1e-5)
    assert torch.allclose(torch.cos(u2_rect_phase), torch.cos(u1_rect), atol=1e-5)
    assert torch.allclose(torch.sin(u2_rect_phase), torch.sin(u1_rect), atol=1e-5)


def test_loop_sin_cos_features_have_inverse_loop_orientation() -> None:
    loops = u2_exp(torch.randn(8, 4))

    features = loop_sin_cos_features(loops)
    inverse_features = loop_sin_cos_features(u2_conj(loops))

    assert torch.allclose(inverse_features[..., :4], -features[..., :4], atol=1e-5)
    assert torch.allclose(inverse_features[..., 4:], features[..., 4:], atol=1e-5)


def test_loop_delta_applies_orientation_only_to_sin_like_terms() -> None:
    transform = FieldTransformation.__new__(FieldTransformation)
    transform.lattice_size = 1
    transform.device = torch.device("cpu")

    loops = u2_exp(torch.randn(1, 2, 1, 1, 4))
    signs = torch.tensor([-1.0, 1.0])

    sin_coeffs = torch.zeros(1, 2, 4, 1, 1)
    sin_coeffs[:, :, 0] = 1.0
    sin_coeffs[:, :, 1] = 1.0
    sin_coeffs = sin_coeffs.reshape(1, 8, 1, 1)

    cos_coeffs = torch.zeros(1, 2, 4, 1, 1)
    cos_coeffs[:, :, 2] = 1.0
    cos_coeffs[:, :, 3] = 1.0
    cos_coeffs = cos_coeffs.reshape(1, 8, 1, 1)

    sin_delta = transform._loop_delta(sin_coeffs, loops, signs)
    sin_delta_flipped = transform._loop_delta(sin_coeffs, loops, -signs)
    cos_delta = transform._loop_delta(cos_coeffs, loops, signs)
    cos_delta_flipped = transform._loop_delta(cos_coeffs, loops, -signs)

    assert torch.allclose(sin_delta_flipped, -sin_delta, atol=1e-5)
    assert torch.allclose(cos_delta_flipped, cos_delta, atol=1e-5)


def test_u2_base_model_returns_phase_only_full_layout_coefficients() -> None:
    model = LocalNet()
    plaq_features = torch.randn(2, 2, 4, 4)
    rect_features = torch.randn(2, 4, 4, 4)

    plaq_coeffs, rect_coeffs = model(plaq_features, rect_features)
    plaq_by_loop = plaq_coeffs.reshape(2, 4, 4, 4, 4)
    rect_by_loop = rect_coeffs.reshape(2, 8, 4, 4, 4)

    assert plaq_coeffs.shape == (2, 16, 4, 4)
    assert rect_coeffs.shape == (2, 32, 4, 4)
    assert torch.allclose(plaq_by_loop[:, :, 1], torch.zeros_like(plaq_by_loop[:, :, 1]))
    assert torch.allclose(plaq_by_loop[:, :, 3], torch.zeros_like(plaq_by_loop[:, :, 3]))
    assert torch.allclose(rect_by_loop[:, :, 1], torch.zeros_like(rect_by_loop[:, :, 1]))
    assert torch.allclose(rect_by_loop[:, :, 3], torch.zeros_like(rect_by_loop[:, :, 3]))


def test_identity_field_has_unit_plaquette_and_zero_action() -> None:
    hmc = HMCU2(lattice_size=4, beta=3.0, n_thermalization_steps=1, n_steps=1, step_size=0.1)
    links = identity_field(4)

    assert torch.allclose(plaquette_mean_from_field(links), torch.tensor(1.0))
    assert torch.allclose(hmc.action(links), torch.tensor(0.0))
    assert torch.allclose(topology_from_field(links), torch.tensor(0.0))


def test_action_matches_beta_over_nc_wilson_normalization() -> None:
    beta = 2.5
    lattice_size = 4
    hmc = HMCU2(lattice_size=lattice_size, beta=beta, n_thermalization_steps=1, n_steps=1, step_size=0.1)
    links = u2_exp(torch.randn(2, lattice_size, lattice_size, 4))
    plaquettes = u2_to_matrix(plaquette_from_field(links))
    wilson_action = 0.5 * beta * torch.sum(2 - torch.diagonal(plaquettes, dim1=-2, dim2=-1).sum(dim=-1).real)

    assert torch.allclose(hmc.action(links), wilson_action, atol=1e-5)


def test_u2_theoretical_plaquette_is_finite_real_value() -> None:
    assert plaq_mean_theory(0.0) == 0.0
    assert 0.0 < plaq_mean_theory(1.0) < 1.0


def test_zero_initialized_field_transform_preserves_links() -> None:
    transform = FieldTransformation(4, n_subsets=1, identity_init=False)
    for model in transform.models:
        for param in model.parameters():
            torch.nn.init.zeros_(param)
    links = u2_exp(torch.randn(2, 2, 4, 4, 4))

    transformed = transform.forward(links)

    assert torch.allclose(u2_to_matrix(transformed), u2_to_matrix(links), atol=1e-5)
    assert torch.allclose(transform.compute_jac_logdet(links), torch.zeros(2), atol=1e-6)


def test_field_transform_inverse_round_trip() -> None:
    set_seed(1234)
    transform = FieldTransformation(4, n_subsets=1, identity_init=True, hyperparams={"init_std": 0.0001})
    links = u2_exp(torch.randn(2, 2, 4, 4, 4))

    transformed = transform.forward(links)
    recovered = transform.inverse(transformed, max_iter=80, tol=1e-5)

    assert torch.allclose(u2_to_matrix(recovered), u2_to_matrix(links), atol=1e-4)


def test_field_transform_jacobian_is_finite_and_differentiable() -> None:
    set_seed(1234)
    transform = FieldTransformation(2, n_subsets=1, identity_init=True, hyperparams={"init_std": 0.0001})
    links = u2_exp(torch.randn(1, 2, 2, 2, 4))

    logdet = transform.compute_jac_logdet(links)
    force = transform.compute_force(links, beta=1.0, transformed=True)

    assert logdet.shape == (1,)
    assert torch.isfinite(logdet).all()
    assert force.shape == (1, 2, 2, 2, 4)
    assert torch.isfinite(force).all()
    assert torch.allclose(u2_log(u2_exp(torch.zeros(4))), torch.zeros(4))


def test_field_transform_manual_jacobian_matches_autograd() -> None:
    set_seed(1234)
    transform = FieldTransformation(
        2,
        n_subsets=1,
        identity_init=True,
        hyperparams={"init_std": 0.0001},
        compile_enabled=False,
    )
    links = u2_exp(0.2 * torch.randn(1, 2, 2, 2, 4))

    manual = transform.compute_jac_logdet(links)
    autograd = transform.compute_jac_logdet_autograd(links)

    assert torch.allclose(manual, autograd, rtol=1e-4, atol=1e-6)


def test_field_transform_full_subset_float32_jacobian_check_tolerance() -> None:
    set_seed(1029)
    transform = FieldTransformation(
        8,
        n_subsets=8,
        identity_init=True,
        hyperparams={"init_std": 0.001},
        compile_enabled=False,
    )
    links = u2_exp(0.2 * torch.randn(1, 2, 8, 8, 4))

    manual = transform.compute_jac_logdet(links)
    autograd = transform.compute_jac_logdet_autograd(links)
    rtol, atol = transform._jacobian_check_tolerances(links)

    assert torch.allclose(manual, autograd, rtol=rtol, atol=atol)


def test_scalar_only_field_transform_is_gauge_covariant() -> None:
    set_seed(1234)
    transform = FieldTransformation(4, n_subsets=8, identity_init=True, hyperparams={"init_std": 0.01})
    links = u2_exp(0.2 * torch.randn(1, 2, 4, 4, 4))
    omega = u2_exp(0.5 * torch.randn(1, 4, 4, 4))

    transformed_gauge_links = transform.forward(gauge_transform_split(links, omega))
    gauge_transformed_output = gauge_transform_split(transform.forward(links), omega)

    assert torch.allclose(
        u2_to_matrix(transformed_gauge_links),
        u2_to_matrix(gauge_transformed_output),
        atol=1e-5,
        rtol=1e-5,
    )


def test_scalar_only_field_transform_logdet_is_gauge_invariant() -> None:
    set_seed(1234)
    transform = FieldTransformation(4, n_subsets=8, identity_init=True, hyperparams={"init_std": 0.01})
    links = u2_exp(0.2 * torch.randn(1, 2, 4, 4, 4))
    omega = u2_exp(0.5 * torch.randn(1, 4, 4, 4))

    logdet = transform.compute_jac_logdet(links)
    gauge_logdet = transform.compute_jac_logdet(gauge_transform_split(links, omega))

    assert torch.allclose(gauge_logdet, logdet, atol=1e-5, rtol=1e-5)


def test_u2_force_matches_finite_difference_directional_derivative() -> None:
    set_seed(1234)
    torch.set_default_dtype(torch.float64)
    try:
        beta = 2.0
        lattice_size = 2
        transform = FieldTransformation(lattice_size, n_subsets=1, identity_init=False)
        links = u2_exp(0.2 * torch.randn(1, 2, lattice_size, lattice_size, 4))
        direction = torch.randn_like(links[..., :4])
        direction = direction / torch.linalg.vector_norm(direction)
        eps = 1e-6

        force = transform.compute_force(links, beta=beta, transformed=False)
        links_plus = u2_mul(u2_exp(eps * direction), links)
        links_minus = u2_mul(u2_exp(-eps * direction), links)
        action_plus = action_from_field_batch(links_plus, beta)
        action_minus = action_from_field_batch(links_minus, beta)
        finite_difference = ((action_plus - action_minus) / (2 * eps)).squeeze(0)
        directional_derivative = torch.sum(force * direction)

        assert torch.allclose(directional_derivative, finite_difference, rtol=1e-5, atol=1e-7)
    finally:
        torch.set_default_dtype(torch.float32)


def test_u2_transformed_force_matches_finite_difference_directional_derivative() -> None:
    set_seed(1234)
    torch.set_default_dtype(torch.float64)
    try:
        beta = 2.0
        lattice_size = 2
        transform = FieldTransformation(
            lattice_size,
            n_subsets=1,
            identity_init=True,
            hyperparams={"init_std": 0.0001},
        )
        links = u2_exp(0.2 * torch.randn(1, 2, lattice_size, lattice_size, 4))
        direction = torch.randn_like(links[..., :4])
        direction = direction / torch.linalg.vector_norm(direction)
        eps = 1e-6

        force = transform.compute_force(links, beta=beta, transformed=True)

        def transformed_potential(varied_links: torch.Tensor) -> torch.Tensor:
            transformed = transform.forward(varied_links)
            return action_from_field_batch(transformed, beta) - transform.compute_jac_logdet(varied_links)

        links_plus = u2_mul(u2_exp(eps * direction), links)
        links_minus = u2_mul(u2_exp(-eps * direction), links)
        finite_difference = (
            (transformed_potential(links_plus) - transformed_potential(links_minus)) / (2 * eps)
        ).squeeze(0)
        directional_derivative = torch.sum(force * direction)

        assert torch.allclose(directional_derivative, finite_difference, rtol=1e-5, atol=1e-7)
    finally:
        torch.set_default_dtype(torch.float32)


def test_u2_transformed_force_decomposition_matches_compute_force() -> None:
    set_seed(1234)
    transform = FieldTransformation(2, n_subsets=1, identity_init=True, hyperparams={"init_std": 0.0001})
    links = u2_exp(0.2 * torch.randn(1, 2, 2, 2, 4))

    force = transform.compute_force(links, beta=2.0, transformed=True)
    total_force, action_force, jac_force, jac_logdet = transform.compute_transformed_force_terms(links, beta=2.0)

    assert torch.allclose(total_force, force, rtol=1e-5, atol=1e-6)
    assert torch.allclose(total_force, action_force - jac_force, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(jac_logdet).all()


def test_hmc_smoke_run_outputs_valid_u2_configs() -> None:
    set_seed(1234)
    hmc = HMCU2(
        lattice_size=4,
        beta=1.0,
        n_thermalization_steps=2,
        n_steps=1,
        step_size=0.05,
        tune_step_size=False,
    )
    links, therm_plaq, therm_acceptance = hmc.thermalize()
    configs, plaq, acceptance, topo, hamiltonians = hmc.run(3, links)
    split_links = torch.stack(configs)
    matrices = u2_to_matrix(split_links)
    identity = torch.eye(2, dtype=matrices.dtype)

    assert split_links.shape == (3, 2, 4, 4, 5)
    assert matrices.shape == (3, 2, 4, 4, 2, 2)
    assert len(therm_plaq) == 2
    assert len(plaq) == 3
    assert len(topo) == 3
    assert len(hamiltonians) == 3
    assert 0 <= therm_acceptance <= 1
    assert 0 <= acceptance <= 1
    assert torch.isfinite(matrices.real).all()
    assert torch.isfinite(matrices.imag).all()
    assert torch.isfinite(torch.tensor(plaq)).all()
    assert torch.allclose(matrices.mH @ matrices, identity.expand_as(matrices), atol=1e-5)
    assert torch.allclose(torch.abs(torch.linalg.det(matrices)), torch.ones(matrices.shape[:-2]), atol=1e-5)


def test_fthmc_smoke_run_outputs_valid_u2_configs() -> None:
    set_seed(1234)
    transform = FieldTransformation(4, n_subsets=1, identity_init=False)
    for model in transform.models:
        for param in model.parameters():
            torch.nn.init.zeros_(param)
    hmc = HMCU2FT(
        lattice_size=4,
        beta=1.0,
        n_thermalization_steps=1,
        n_steps=1,
        step_size=0.02,
        field_transformation=transform.field_transformation,
        compute_jac_logdet=transform.compute_jac_logdet,
        tune_step_size=False,
    )
    links, therm_plaq, therm_acceptance = hmc.thermalize()
    configs, plaq, acceptance, topo, hamiltonians = hmc.run(2, links, save_config=True)
    split_links = torch.stack(configs)
    matrices = u2_to_matrix(split_links)
    identity = torch.eye(2, dtype=matrices.dtype)

    assert split_links.shape == (2, 2, 4, 4, 5)
    assert len(therm_plaq) == 1
    assert len(plaq) == 2
    assert len(topo) == 2
    assert len(hamiltonians) == 2
    assert 0 <= therm_acceptance <= 1
    assert 0 <= acceptance <= 1
    assert torch.isfinite(matrices.real).all()
    assert torch.isfinite(matrices.imag).all()
    assert torch.isfinite(torch.tensor(plaq)).all()
    assert torch.allclose(matrices.mH @ matrices, identity.expand_as(matrices), atol=1e-5)
