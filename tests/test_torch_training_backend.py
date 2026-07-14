import tempfile

import numpy as np
import torch

from nthmc.core.training import fixed_batches
from nthmc.u1.training import FieldTransformation as TorchU1
from nthmc.u2.training import FieldTransformation as TorchU2
from nthmc.u2.training_observables import identity_field
from nthmc.u1.field_transform import FieldTransformation as JaxU1
from nthmc.u2.field_transform import FieldTransformation as JaxU2


def test_torch_identity_inverse_stops_after_one_iteration() -> None:
    u1 = TorchU1(2, n_subsets=1, model_tag="base", hyperparams={"inverse_max_iters": 5})
    theta = torch.zeros(2, 2, 2, 2)
    _, u1_diag = u1.inverse(theta, return_diagnostics=True)

    u2 = TorchU2(2, n_subsets=1, model_tag="base", hyperparams={"inverse_max_iters": 5})
    links = identity_field(2).unsqueeze(0).repeat(2, 1, 1, 1, 1)
    _, u2_diag = u2.inverse(links, return_diagnostics=True)

    assert u1_diag["max_iterations"] == 1
    assert u2_diag["max_iterations"] == 1
    assert u1_diag["n_not_converged"] == 0
    assert u2_diag["n_not_converged"] == 0


def test_torch_inverse_nonconvergence_stops_at_max_iterations() -> None:
    transform = TorchU1(2, n_subsets=1, model_tag="base")
    theta = torch.zeros(2, 2, 2, 2)
    _, diagnostics = transform.inverse(theta, max_iter=3, tol=0.0, return_diagnostics=True)
    assert diagnostics["max_iterations"] == 3
    assert diagnostics["n_not_converged"] == 2


def test_fixed_batches_and_masked_u1_gradient_match_unpadded() -> None:
    transform = TorchU1(2, n_subsets=1, model_tag="base", hyperparams={"inverse_max_iters": 2})
    transform.train_beta = 1.0
    data = 0.05 * torch.arange(3 * 2 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 2)
    padded, mask = fixed_batches(data, 2, shuffle=False, seed=0)[-1]

    loss_masked = transform.loss_fn(padded, mask)
    grads_masked = torch.autograd.grad(loss_masked, [p for model in transform.models for p in model.parameters()])
    loss_single = transform.loss_fn(padded[:1])
    grads_single = torch.autograd.grad(loss_single, [p for model in transform.models for p in model.parameters()])

    assert torch.allclose(loss_masked, loss_single)
    assert all(torch.allclose(left, right, atol=1e-6, rtol=1e-5) for left, right in zip(grads_masked, grads_single))


def test_torch_checkpoint_exports_loadable_jax_params() -> None:
    for torch_cls, jax_cls in ((TorchU1, JaxU1), (TorchU2, JaxU2)):
        with tempfile.TemporaryDirectory() as directory:
            torch_transform = torch_cls(2, n_subsets=1, model_tag="base", model_dir=directory, save_tag="test")
            torch_transform.train_beta = 1.0
            with torch.no_grad():
                torch_transform.models[0].out_scale.scale.fill_(0.1)
                torch_transform.models[0].conv_output.bias.add_(0.05)
            torch_transform.save_best_model(0, 1.25)

            jax_transform = jax_cls(2, n_subsets=1, model_tag="base", model_dir=directory, save_tag="test")
            jax_transform.load_best_model(1.0)
            torch_weight = torch_transform.models[0].conv_input.weight.detach().numpy()
            jax_weight = np.asarray(jax_transform.params["subsets"][0]["conv_input"]["weight"])
            assert np.array_equal(torch_weight, jax_weight)
            sample = (
                0.01 * torch.randn(1, 2, 2, 2)
                if torch_cls is TorchU1
                else identity_field(2).unsqueeze(0)
            )
            torch_output = torch_transform.forward(sample).detach().numpy()
            jax_output = np.asarray(jax_transform.forward(sample.numpy()))
            assert np.allclose(torch_output, jax_output, atol=1e-6, rtol=1e-6)
            torch_logdet = torch_transform.compute_jac_logdet(sample).detach().numpy()
            jax_logdet = np.asarray(jax_transform.compute_jac_logdet(sample.numpy()))
            assert np.allclose(torch_logdet, jax_logdet, atol=2e-5, rtol=2e-5)
