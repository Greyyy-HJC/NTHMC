import tempfile

import numpy as np
import torch

from nthmc.core.checkpoint import load_checkpoint
from nthmc.core.training import fixed_batches, global_batch_size, local_batch, models_to_jax_params
from nthmc.u1.training import FieldTransformation as TorchU1
from nthmc.u2.training import FieldTransformation as TorchU2


def test_torch_inverse_nonconvergence_stops_at_max_iterations() -> None:
    transform = TorchU1(2, n_subsets=1, model_tag="base")
    theta = torch.zeros(2, 2, 2, 2)
    _, diagnostics = transform.inverse(theta, max_iter=3, tol=0.0, return_diagnostics=True)
    assert diagnostics["max_iterations"] == 3
    assert diagnostics["n_not_converged"] == 2


def test_torch_models_use_default_convolution_init_and_zero_output_gate() -> None:
    for transform_cls in (TorchU1, TorchU2):
        transform = transform_cls(2, n_subsets=1, model_tag="base")
        model = transform.models[0]

        assert model.conv_input.weight.std() > 0.005
        assert model.conv_output.weight.std() > 0.005
        assert torch.count_nonzero(model.out_scale.scale) == 0


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


def test_distributed_batches_use_per_rank_batch_size() -> None:
    per_rank_batch_size = 12
    world_size = 4
    distributed_batch_size = global_batch_size(per_rank_batch_size, world_size)
    data = torch.arange(3276)
    batches = fixed_batches(data, distributed_batch_size, shuffle=False, seed=0)

    assert distributed_batch_size == 48
    assert len(batches) == 69
    for rank in range(world_size):
        local, local_mask = local_batch(*batches[0], rank=rank, world_size=world_size)
        assert len(local) == per_rank_batch_size
        assert local_mask.all()


def test_single_rank_batch_size_is_unchanged() -> None:
    assert global_batch_size(12, 1) == 12


def test_torch_checkpoint_exports_loadable_jax_params() -> None:
    for torch_cls in (TorchU1, TorchU2):
        with tempfile.TemporaryDirectory() as directory:
            torch_transform = torch_cls(2, n_subsets=1, model_tag="base", model_dir=directory, save_tag="test")
            torch_transform.train_beta = 1.0
            with torch.no_grad():
                torch_transform.models[0].out_scale.scale.fill_(0.1)
                torch_transform.models[0].conv_output.bias.add_(0.05)
            torch_transform.save_best_model(0, 1.25)

            template = models_to_jax_params(list(torch_transform.models))
            loaded, metadata = load_checkpoint(torch_transform.jax_checkpoint_path(1.0), template)

            assert metadata["model_tag"] == "base"
            assert metadata["param_count"] == 5
            assert np.array_equal(
                torch_transform.models[0].conv_input.weight.detach().numpy(),
                np.asarray(loaded["subsets"][0]["conv_input"]["weight"]),
            )
