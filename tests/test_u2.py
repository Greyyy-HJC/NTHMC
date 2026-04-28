import torch

from nthmc.u2.field_transform import FieldTransformation
from nthmc.u2.u2_fthmc import HMCU2FT
from nthmc.u2.u2_hmc import HMCU2
from nthmc.u2.u2_observables import (
    identity_field,
    matrix_to_u2,
    plaq_mean_theory,
    plaquette_from_field_batch,
    plaquette_mean_from_field,
    plaquette_from_field,
    set_seed,
    topology_from_field,
    u2_exp,
    u2_mul,
    u2_to_matrix,
)


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
    transform = FieldTransformation(4, n_subsets=8, identity_init=False)
    for model in transform.models:
        for param in model.parameters():
            torch.nn.init.zeros_(param)
    links = u2_exp(torch.randn(2, 2, 4, 4, 4))

    transformed = transform.forward(links)

    assert torch.allclose(u2_to_matrix(transformed), u2_to_matrix(links), atol=1e-5)
    assert torch.allclose(transform.compute_jac_logdet(links), torch.zeros(2))


def test_field_transform_inverse_round_trip() -> None:
    set_seed(1234)
    transform = FieldTransformation(4, n_subsets=8, identity_init=True, hyperparams={"init_std": 0.001})
    links = u2_exp(torch.randn(2, 2, 4, 4, 4))

    transformed = transform.forward(links)
    recovered = transform.inverse(transformed)

    assert torch.allclose(u2_to_matrix(recovered), u2_to_matrix(links), atol=1e-5)


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
    transform = FieldTransformation(4, n_subsets=8, identity_init=False)
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
