"""VICReg-лосс: инвариантность + variance + covariance.

Зачем: без регуляризации JEPA тривиально коллапсирует — энкодер выучивает
константу, и предсказание становится идеальным, но бесполезным.
VICReg решает это явно: variance-штраф заставляет каждую координату латента
иметь std >= 1 по батчу, covariance-штраф декоррелирует координаты.

Коэффициенты 25 / 25 / 1 — стандарт из статьи (Bardes et al., 2022).
"""
import torch
import torch.nn.functional as F


def variance_term(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(1.0 - std).mean()


def covariance_term(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / d


def vicreg_loss(z_pred: torch.Tensor, z_target: torch.Tensor,
                sim_w: float = 25.0, var_w: float = 25.0, cov_w: float = 1.0):
    """Возвращает (total, dict с компонентами для логирования)."""
    sim = F.mse_loss(z_pred, z_target)
    var = variance_term(z_pred) + variance_term(z_target)
    cov = covariance_term(z_pred) + covariance_term(z_target)
    total = sim_w * sim + var_w * var + cov_w * cov
    return total, {"sim": sim.item(), "var": var.item(), "cov": cov.item()}
