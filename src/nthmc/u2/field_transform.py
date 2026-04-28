"""Base neural field transformation for 2D U(2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from nthmc.u2.models import choose_model
from nthmc.u2.u2_observables import (
    action_from_field_batch,
    format_beta,
    get_link_mask,
    identity_like,
    plaquette_from_field_batch,
    u2_exp,
    u2_mul,
    u2_normalize,
)


class FieldTransformation:
    """Volume-preserving U(2) coupling transformation for FT-HMC."""

    def __init__(
        self,
        lattice_size: int,
        *,
        device: str = "cpu",
        n_subsets: int = 8,
        if_check_jac: bool = False,
        num_workers: int = 0,
        identity_init: bool = True,
        model_tag: str = "base",
        save_tag: str | None = None,
        model_dir: str | Path = "artifacts/models",
        plot_dir: str | Path = "plots",
        dump_dir: str | Path = "dumps",
        hyperparams: dict[str, float] | None = None,
        fabric=None,
    ) -> None:
        self.lattice_size = lattice_size
        self.device = torch.device(device)
        self.n_subsets = n_subsets
        self.if_check_jac = if_check_jac
        self.num_workers = num_workers
        self.model_tag = model_tag
        self.save_tag = save_tag or "base"
        self.model_dir = Path(model_dir)
        self.plot_dir = Path(plot_dir)
        self.dump_dir = Path(dump_dir)
        self.train_beta: float | None = None
        self.fabric = fabric
        self.print = fabric.print if fabric is not None else print
        self.backward = fabric.backward if fabric is not None else torch.autograd.backward

        self.hyperparams = {
            "init_std": 0.001,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "factor": 0.5,
            "patience": 5,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)

        model_cls = choose_model(model_tag)
        raw_models = nn.ModuleList([model_cls().to(self.device) for _ in range(n_subsets)])
        if identity_init:
            for model in raw_models:
                for param in model.parameters():
                    nn.init.normal_(param, mean=0.0, std=self.hyperparams["init_std"])

        raw_optimizers = [
            torch.optim.AdamW(
                model.parameters(),
                lr=self.hyperparams["lr"],
                weight_decay=self.hyperparams["weight_decay"],
            )
            for model in raw_models
        ]
        self.models = []
        self.optimizers = []
        for model, optimizer in zip(raw_models, raw_optimizers):
            if self.fabric is not None:
                model, optimizer = self.fabric.setup(model, optimizer)
            self.models.append(model)
            self.optimizers.append(optimizer)
        self.schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hyperparams["factor"],
                patience=int(self.hyperparams["patience"]),
            )
            for optimizer in self.optimizers
        ]

    def _masked_links_for_features(self, links: torch.Tensor, index: int) -> torch.Tensor:
        mask = get_link_mask(index, links.shape[0], self.lattice_size, self.device)
        return torch.where(mask, identity_like(links), links)

    def compute_delta(self, links: torch.Tensor, index: int) -> torch.Tensor:
        feature_links = self._masked_links_for_features(links, index)
        plaquettes = plaquette_from_field_batch(feature_links)
        features = torch.cat(
            [
                torch.sin(plaquettes[..., :1]),
                torch.cos(plaquettes[..., :1]),
                plaquettes[..., 1:],
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)
        delta = self.models[index](features)
        mask = get_link_mask(index, links.shape[0], self.lattice_size, self.device)
        return delta[:, None] * mask.to(delta.dtype)

    def ft_phase(self, links: torch.Tensor, index: int) -> torch.Tensor:
        delta = self.compute_delta(links, index)
        return u2_mul(u2_exp(delta), links)

    def inverse_phase(self, links: torch.Tensor, index: int) -> torch.Tensor:
        delta = self.compute_delta(links, index)
        return u2_mul(u2_exp(-delta), links)

    def forward(self, links: torch.Tensor) -> torch.Tensor:
        links_curr = u2_normalize(links)
        for index in range(self.n_subsets):
            links_curr = self.ft_phase(links_curr, index)
        return u2_normalize(links_curr)

    def field_transformation(self, links: torch.Tensor) -> torch.Tensor:
        return self.forward(links.unsqueeze(0)).squeeze(0)

    def inverse(self, links: torch.Tensor) -> torch.Tensor:
        links_curr = u2_normalize(links)
        for index in reversed(range(self.n_subsets)):
            links_curr = self.inverse_phase(links_curr, index)
        return u2_normalize(links_curr)

    def compute_jac_logdet(self, links: torch.Tensor) -> torch.Tensor:
        return torch.zeros(links.shape[0], device=self.device, dtype=links.dtype)

    def compute_action(self, links: torch.Tensor, beta: float) -> torch.Tensor:
        return action_from_field_batch(links, beta)

    def compute_force(self, links: torch.Tensor, beta: float, *, transformed: bool = False) -> torch.Tensor:
        algebra = torch.zeros(
            (*links.shape[:-1], 4),
            device=self.device,
            dtype=links.dtype,
            requires_grad=True,
        )
        varied_links = u2_mul(u2_exp(algebra), links.detach())
        if transformed:
            links_ori = self.forward(varied_links)
            total_action = self.compute_action(links_ori, beta) - self.compute_jac_logdet(varied_links)
        else:
            total_action = self.compute_action(varied_links, beta)
        return torch.autograd.grad(total_action.sum(), algebra, create_graph=True)[0]

    def loss_fn(self, links_ori: torch.Tensor) -> torch.Tensor:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        links_new = self.inverse(links_ori)
        force_new = self.compute_force(links_new, self.train_beta, transformed=True)
        volume = self.lattice_size * self.lattice_size
        return (
            torch.norm(force_new, p=2) / (volume**0.5)
            + torch.norm(force_new, p=4) / (volume**0.25)
            + torch.norm(force_new, p=6) / (volume ** (1 / 6))
            + torch.norm(force_new, p=8) / (volume ** (1 / 8))
        )

    def train_step(self, links_ori: torch.Tensor) -> float:
        links_ori = links_ori.to(self.device)
        loss = self.loss_fn(links_ori)
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        self.backward(loss)
        for optimizer in self.optimizers:
            optimizer.step()
        return float(loss.detach().cpu())

    def evaluate_step(self, links_ori: torch.Tensor) -> float:
        links_ori = links_ori.to(self.device)
        loss = self.loss_fn(links_ori)
        return float(loss.detach().cpu())

    def train(self, train_data: torch.Tensor, test_data: torch.Tensor, train_beta: float, *, n_epochs: int, batch_size: int) -> None:
        self.train_beta = train_beta
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=self.num_workers)
        if self.fabric is not None:
            train_loader = self.fabric.setup_dataloaders(train_loader)
            test_loader = self.fabric.setup_dataloaders(test_loader)

        train_losses: list[float] = []
        test_losses: list[float] = []
        best_loss = float("inf")

        for epoch in tqdm(range(n_epochs), desc="Training epochs"):
            self._set_models_mode(True)
            epoch_losses = [self.train_step(batch) for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")]
            train_loss = float(np.mean(epoch_losses))
            train_losses.append(train_loss)

            self._set_models_mode(False)
            test_epoch_losses = [self.evaluate_step(batch) for batch in tqdm(test_loader, desc="Evaluating")]
            test_loss = float(np.mean(test_epoch_losses))
            test_losses.append(test_loss)

            self.print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.6f} - Test Loss: {test_loss:.6f}")
            if test_loss < best_loss:
                self.save_best_model(epoch, test_loss)
                best_loss = test_loss
            for scheduler in self.schedulers:
                scheduler.step(test_loss)

        self.plot_training_history(train_losses, test_losses)
        if self.fabric is not None:
            self.fabric.barrier()
        self.load_best_model(train_beta)

    def _set_models_mode(self, is_train: bool) -> None:
        for model in self.models:
            model.train() if is_train else model.eval()

    def checkpoint_path(self, train_beta: float) -> Path:
        return self.model_dir / f"best_model_train_beta{format_beta(train_beta)}_{self.save_tag}.pt"

    def save_best_model(self, epoch: int, loss: float) -> None:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        if self.fabric is not None and self.fabric.global_rank != 0:
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        save_dict = {"epoch": epoch, "loss": loss}
        for index, model in enumerate(self.models):
            save_dict[f"model_state_dict_{index}"] = model.state_dict()
        for index, optimizer in enumerate(self.optimizers):
            save_dict[f"optimizer_state_dict_{index}"] = optimizer.state_dict()
        torch.save(save_dict, self.checkpoint_path(self.train_beta))

    def load_best_model(self, train_beta: float) -> None:
        checkpoint = torch.load(self.checkpoint_path(train_beta), map_location=self.device, weights_only=False)
        for index, model in enumerate(self.models):
            state_dict = checkpoint[f"model_state_dict_{index}"]
            if any(key.startswith("module.") for key in state_dict):
                state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
        self.print(f"Loaded best model from epoch {checkpoint['epoch'] + 1} with loss {checkpoint['loss']:.6f}")

    def plot_training_history(self, train_losses: list[float], test_losses: list[float]) -> None:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        if self.fabric is not None and self.fabric.global_rank != 0:
            return
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        beta_tag = format_beta(self.train_beta)

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train")
        plt.plot(test_losses, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"cnn_loss_train_beta{beta_tag}_{self.save_tag}.pdf", transparent=True)
        plt.close()

        np.savetxt(self.dump_dir / f"train_loss_train_beta{beta_tag}_{self.save_tag}.csv", train_losses, fmt="%.6e")
        np.savetxt(self.dump_dir / f"test_loss_train_beta{beta_tag}_{self.save_tag}.csv", test_losses, fmt="%.6e")
