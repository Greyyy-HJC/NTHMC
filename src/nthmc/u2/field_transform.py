"""JAX field transformation scaffold for 2D U(2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nthmc.core.checkpoint import load_checkpoint, save_checkpoint
from nthmc.u2.models import init_transform_params
from nthmc.u2.u2_observables import action_from_field, force_from_field, format_beta, u2_normalize

Array = Any
Params = dict[str, Any]


class FieldTransformation:
    """JAX U(2) identity transform with JAX checkpoint/training plumbing.

    This keeps the first JAX U(2) FT-HMC path mathematically exact:
    forward/inverse are identity and the Jacobian log determinant is zero. The
    standard U(2) Wilson force and HMC path are fully JAX and JIT-compatible.
    """

    def __init__(
        self,
        lattice_size: int,
        *,
        device: str = "cpu",
        n_subsets: int = 8,
        if_check_jac: bool = False,
        num_workers: int = 0,
        model_tag: str = "base",
        save_tag: str | None = None,
        model_dir: str | Path = "artifacts/models",
        plot_dir: str | Path = "plots",
        dump_dir: str | Path = "dumps",
        hyperparams: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.lattice_size = lattice_size
        self.device = device
        self.n_subsets = n_subsets
        self.if_check_jac = if_check_jac
        self.num_workers = num_workers
        self.model_tag = model_tag
        self.save_tag = save_tag or "opt"
        self.model_dir = Path(model_dir)
        self.plot_dir = Path(plot_dir)
        self.dump_dir = Path(dump_dir)
        self.train_beta: float | None = None
        self.hyperparams: dict[str, Any] = {
            "init_std": 0.001,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "max_grad_norm": 10.0,
            "early_stop_patience": 20,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)
        self.params = init_transform_params(jax.random.PRNGKey(0), model_tag, n_subsets, init_std=float(self.hyperparams["init_std"]))
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(float(self.hyperparams["max_grad_norm"])),
            optax.adamw(float(self.hyperparams["lr"]), weight_decay=float(self.hyperparams["weight_decay"])),
        )
        self.opt_state = self.optimizer.init(self.params)

    def _checkpoint_template(self) -> Params:
        return init_transform_params(jax.random.PRNGKey(0), self.model_tag, self.n_subsets, init_std=float(self.hyperparams["init_std"]))

    def checkpoint_path(self, train_beta: float) -> Path:
        return self.model_dir / f"best_model_train_beta{format_beta(train_beta)}_{self.save_tag}.npz"

    def save_best_model(self, train_beta: float, epoch: int, loss: float) -> None:
        metadata = {
            "system": "2du2",
            "transform": "identity",
            "model_tag": self.model_tag,
            "n_subsets": self.n_subsets,
            "lattice_size": self.lattice_size,
            "train_beta": float(train_beta),
            "epoch": int(epoch),
            "loss": float(loss),
            "hyperparams": self.hyperparams,
        }
        save_checkpoint(self.checkpoint_path(train_beta), params=self.params, opt_state=self.opt_state, metadata=metadata)

    def load_best_model(self, train_beta: float) -> None:
        self.params, metadata = load_checkpoint(self.checkpoint_path(train_beta), self._checkpoint_template())
        self.opt_state = self.optimizer.init(self.params)
        print(f"Loaded JAX U(2) checkpoint from epoch {metadata.get('epoch')} with loss {metadata.get('loss')}")

    def freeze_models_for_eval(self) -> None:
        return None

    def enable_eval_compile(self, *, backend: str = "jax") -> None:
        return None

    def forward(self, links: Array) -> Array:
        return u2_normalize(links)

    def field_transformation(self, links: Array) -> Array:
        return self.forward(jnp.asarray(links))

    def field_transformation_compiled(self, links: Array) -> Array:
        return self.field_transformation(links)

    def inverse(self, links: Array, **_: Any) -> Array:
        return u2_normalize(links)

    def inverse_field_transformation(self, links: Array) -> Array:
        return self.inverse(links)

    def compute_jac_logdet(self, links: Array) -> Array:
        return jnp.zeros((links.shape[0],), dtype=links.dtype)

    def compute_jac_logdet_compiled(self, links: Array) -> Array:
        return self.compute_jac_logdet(links)

    def compute_force(self, links: Array, beta: float, *, transformed: bool = False) -> Array:
        return jax.vmap(lambda x: force_from_field(x, beta))(jnp.asarray(links))

    def loss_fn_with_params(self, params: Params, links: Array, beta: float) -> Array:
        del params
        force = jax.vmap(lambda x: force_from_field(x, beta))(links)
        volume = self.lattice_size * self.lattice_size
        flat = force.reshape(force.shape[0], -1)
        return jnp.mean(
            jnp.linalg.norm(flat, ord=2, axis=1) / (volume**0.5)
            + jnp.linalg.norm(flat, ord=4, axis=1) / (volume**0.25)
            + jnp.linalg.norm(flat, ord=6, axis=1) / (volume ** (1 / 6))
            + jnp.linalg.norm(flat, ord=8, axis=1) / (volume ** (1 / 8))
        )

    def loss_fn(self, links: Array) -> Array:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        return self.loss_fn_with_params(self.params, jnp.asarray(links), self.train_beta)

    def _batches(self, data: np.ndarray, batch_size: int, rng: np.random.Generator, *, shuffle: bool) -> list[np.ndarray]:
        indices = np.arange(len(data))
        if shuffle:
            rng.shuffle(indices)
        return [data[indices[start : start + batch_size]] for start in range(0, len(indices), batch_size)]

    def train(self, train_data: Array, test_data: Array, train_beta: float, *, n_epochs: int, batch_size: int) -> None:
        self.train_beta = float(train_beta)
        train_np = np.asarray(train_data, dtype=np.float32)
        test_np = np.asarray(test_data, dtype=np.float32)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        eval_step = jax.jit(lambda batch: self.loss_fn_with_params(self.params, batch, self.train_beta))
        rng = np.random.default_rng(0)
        train_losses: list[float] = []
        test_losses: list[float] = []
        best_loss = float("inf")
        for epoch in tqdm(range(n_epochs), desc="Training epochs"):
            train_loss = float(np.mean([float(eval_step(jnp.asarray(batch))) for batch in self._batches(train_np, batch_size, rng, shuffle=True)]))
            eval_losses = [float(eval_step(jnp.asarray(batch))) for batch in self._batches(test_np, batch_size, rng, shuffle=False)]
            test_loss = float(np.mean(eval_losses)) if eval_losses else train_loss
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch + 1}/{n_epochs}: train_loss={train_loss:.6f} test_loss={test_loss:.6f}")
            if test_loss < best_loss:
                best_loss = test_loss
                self.save_best_model(train_beta, epoch, test_loss)
        tag = f"train_beta{format_beta(train_beta)}_{self.save_tag}"
        np.savetxt(self.dump_dir / f"train_loss_{tag}.csv", np.asarray(train_losses), fmt="%.8e")
        np.savetxt(self.dump_dir / f"test_loss_{tag}.csv", np.asarray(test_losses), fmt="%.8e")
        fig, ax = plt.subplots()
        ax.plot(train_losses, label="train")
        ax.plot(test_losses, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plot_dir / f"cnn_loss_{tag}.pdf", transparent=True)
        plt.close(fig)
