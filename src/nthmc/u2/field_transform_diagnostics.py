"""Diagnostics helpers for U(2) field-transformation training."""

from __future__ import annotations

from typing import Any

import torch

from nthmc.u2.u2_observables import (
    get_link_mask,
    plaquette_from_field_batch,
    rectangle_from_field_batch,
    u2_conj,
    u2_exp,
    u2_log,
    u2_mul,
    u2_normalize,
)


def _parameter_diagnostics(transform: Any) -> tuple[float, float]:
    norm_sq = 0.0
    max_abs = 0.0
    for model in transform.models:
        for param in model.parameters():
            detached = param.detach()
            norm_sq += float(torch.sum(detached**2).cpu())
            max_abs = max(max_abs, float(torch.max(torch.abs(detached)).cpu()))
    return norm_sq**0.5, max_abs


def _delta_diagnostics(transform: Any, links: torch.Tensor) -> tuple[float, float]:
    volume = transform.lattice_size * transform.lattice_size
    links_curr = u2_normalize(links)
    delta_norms = []
    for index in range(transform.n_subsets):
        delta = transform.compute_delta(links_curr, index)
        delta_norms.append(torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=1) / (volume**0.5))
        links_curr = u2_mul(u2_exp(delta), links_curr)
    if not delta_norms:
        return 0.0, 0.0
    stacked = torch.stack(delta_norms, dim=1)
    return float(stacked.mean().cpu()), float(stacked.max().cpu())


def _coefficient_diagnostics(transform: Any, links: torch.Tensor) -> dict[str, float]:
    plaq_coeff_abs = []
    rect_coeff_abs = []
    links_curr = u2_normalize(links)
    for index in range(transform.n_subsets):
        plaq = plaquette_from_field_batch(links_curr)
        rect = rectangle_from_field_batch(links_curr)
        plaq_loops = transform._plaq_loop_stack(links_curr)
        rect_loops = transform._rect_loop_stack(links_curr)
        plaq_coeffs, rect_coeffs = transform.compute_coefficients(links_curr, index, plaq, rect)
        plaq_coeff_abs.append(torch.abs(plaq_coeffs).reshape(plaq_coeffs.shape[0], -1))
        rect_coeff_abs.append(torch.abs(rect_coeffs).reshape(rect_coeffs.shape[0], -1))
        delta = transform._plaq_delta(plaq_coeffs, plaq_loops) + transform._rect_delta(rect_coeffs, rect_loops)
        mask = get_link_mask(index, links.shape[0], transform.lattice_size, transform.device)
        links_curr = u2_mul(u2_exp(delta * mask.to(delta.dtype)), links_curr)
    if not plaq_coeff_abs or not rect_coeff_abs:
        return {
            "k0_abs_mean": 0.0,
            "k0_abs_max": 0.0,
            "k0_sat_frac": 0.0,
            "k1_abs_mean": 0.0,
            "k1_abs_max": 0.0,
            "k1_sat_frac": 0.0,
        }
    k0_all = torch.cat(plaq_coeff_abs, dim=1)
    k1_all = torch.cat(rect_coeff_abs, dim=1)
    k0_cap = 1 / 6
    k1_cap = 1 / 45
    return {
        "k0_abs_mean": float(k0_all.mean().cpu()),
        "k0_abs_max": float(k0_all.max().cpu()),
        "k0_sat_frac": float((k0_all > 0.95 * k0_cap).to(torch.float32).mean().cpu()),
        "k1_abs_mean": float(k1_all.mean().cpu()),
        "k1_abs_max": float(k1_all.max().cpu()),
        "k1_sat_frac": float((k1_all > 0.95 * k1_cap).to(torch.float32).mean().cpu()),
    }


def collect_training_diagnostics(transform: Any, test_data: torch.Tensor, batch_size: int, grad_norm: float) -> dict[str, Any] | None:
    if transform.fabric is not None and transform.fabric.global_rank != 0:
        return None
    n = min(8, int(test_data.shape[0]), int(batch_size))
    if n <= 0:
        return None

    probe = test_data[:n].to(transform.device)
    with torch.no_grad():
        inv, inverse_diag = transform.inverse(probe, return_diagnostics=True)
        recon = transform.forward(inv)
        x0 = u2_normalize(probe)
        rt = u2_mul(recon, u2_conj(x0))
        round_trip_mean_log_norm = torch.linalg.norm(u2_log(rt).reshape(probe.shape[0], -1), dim=1).mean().item()
        delta_norm_mean, delta_norm_max = _delta_diagnostics(transform, inv)
        coefficient_diag = _coefficient_diagnostics(transform, inv)
        param_norm, param_max = _parameter_diagnostics(transform)

    with torch.enable_grad():
        force, action_force, jac_force, jac_logdet, topo_grad = transform.compute_transformed_force_terms(
            inv.detach(),
            transform.train_beta,
            create_graph=False,
            include_topo_grad=True,
        )
        force_components = transform._force_loss_components(force)
        action_force_components = transform._force_loss_components(action_force)
        jac_force_components = transform._force_loss_components(jac_force)
        force_topo_alignment = transform._force_topology_alignment(force, topo_grad)

    force_weighted_loss = transform._weighted_force_loss(force_components)
    jac_logdet = jac_logdet.detach()
    jac_logdet_diag = {
        "mean": jac_logdet.mean().item(),
        "std": jac_logdet.std(unbiased=False).item(),
        "min": jac_logdet.min().item(),
        "max": jac_logdet.max().item(),
    }
    return {
        "grad_norm": grad_norm,
        "inverse_diag": inverse_diag,
        "round_trip_mean_log_norm": round_trip_mean_log_norm,
        "delta_norm_mean": delta_norm_mean,
        "delta_norm_max": delta_norm_max,
        "coefficient_diag": coefficient_diag,
        "param_norm": param_norm,
        "param_max": param_max,
        "force_weighted_loss": force_weighted_loss,
        "force_components": force_components,
        "action_force_components": action_force_components,
        "jac_force_components": jac_force_components,
        "force_topo_alignment": force_topo_alignment,
        "jac_logdet_diag": jac_logdet_diag,
    }


def print_training_diagnostics(transform: Any, diagnostics: dict[str, Any], epoch_display: int, n_epochs: int) -> None:
    inverse_diag = diagnostics["inverse_diag"]
    coefficient_diag = diagnostics["coefficient_diag"]
    jac_logdet_diag = diagnostics["jac_logdet_diag"]
    force_components = diagnostics["force_components"]
    action_force_components = diagnostics["action_force_components"]
    jac_force_components = diagnostics["jac_force_components"]

    transform.print(
        f"Epoch {epoch_display}/{n_epochs} inverse_diag: "
        f"max_final_diff={inverse_diag['max_final_diff']:.2e} "
        f"mean_final_diff={inverse_diag['mean_final_diff']:.2e} "
        f"n_subsets_not_converged={inverse_diag['n_not_converged']} "
        f"round_trip_mean_log_norm={diagnostics['round_trip_mean_log_norm']:.2e}"
    )
    transform.print(
        f"Epoch {epoch_display}/{n_epochs} train_diag: "
        f"grad_norm_pre_clip={diagnostics['grad_norm']:.2e} "
        f"param_norm={diagnostics['param_norm']:.2e} param_max_abs={diagnostics['param_max']:.2e} "
        f"delta_norm_mean={diagnostics['delta_norm_mean']:.2e} delta_norm_max={diagnostics['delta_norm_max']:.2e} "
        f"loss_weights={transform._loss_weights()} "
        f"force_weighted_loss={diagnostics['force_weighted_loss']:.6f} "
        f"k0_abs_mean={coefficient_diag['k0_abs_mean']:.2e} "
        f"k0_abs_max={coefficient_diag['k0_abs_max']:.2e} "
        f"k0_sat_frac={coefficient_diag['k0_sat_frac']:.3f} "
        f"k1_abs_mean={coefficient_diag['k1_abs_mean']:.2e} "
        f"k1_abs_max={coefficient_diag['k1_abs_max']:.2e} "
        f"k1_sat_frac={coefficient_diag['k1_sat_frac']:.3f} "
        f"jac_logdet_mean={jac_logdet_diag['mean']:.2e} "
        f"jac_logdet_std={jac_logdet_diag['std']:.2e} "
        f"jac_logdet_min={jac_logdet_diag['min']:.2e} "
        f"jac_logdet_max={jac_logdet_diag['max']:.2e} "
        f"force_l2={force_components['l2']:.6f} "
        f"force_l4={force_components['l4']:.6f} "
        f"force_l6={force_components['l6']:.6f} "
        f"force_l8={force_components['l8']:.6f} "
        f"force_topo_cos={diagnostics['force_topo_alignment']:.6f}"
    )
    transform.print(
        f"Epoch {epoch_display}/{n_epochs} force_split_diag: "
        f"action_l2={action_force_components['l2']:.6f} "
        f"action_l4={action_force_components['l4']:.6f} "
        f"action_l6={action_force_components['l6']:.6f} "
        f"action_l8={action_force_components['l8']:.6f} "
        f"jac_l2={jac_force_components['l2']:.6f} "
        f"jac_l4={jac_force_components['l4']:.6f} "
        f"jac_l6={jac_force_components['l6']:.6f} "
        f"jac_l8={jac_force_components['l8']:.6f}"
    )


def maybe_log_training_diagnostics(
    transform: Any,
    test_data: torch.Tensor,
    batch_size: int,
    epoch_display: int,
    n_epochs: int,
    grad_norm: float,
) -> None:
    diagnostics = collect_training_diagnostics(transform, test_data, batch_size, grad_norm)
    if diagnostics is None:
        return
    print_training_diagnostics(transform, diagnostics, epoch_display, n_epochs)

