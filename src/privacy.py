"""
privacy.py
==========
Implementation de la confidentialite differentielle appliquee a
l'entrainement (DP-SGD, Abadi et al., 2016), exactement l'Algorithme 5.1 et
l'equation 5.2 du memoire :

    g_hat_i = g_i / max(1, ||g_i||_2 / C)                         (ecretage)
    g_tilde = (1/L) * ( sum_i g_hat_i + N(0, sigma^2 C^2 I) )     (bruitage)

et une estimation simplifiee du budget de confidentialite cumule (Eq. 5.6,
"moments accountant") :

    epsilon_total ~= O( q / sigma * sqrt(T * log(1/delta)) )

AVERTISSEMENT (transparence scientifique, cf. chapitre 5.1.3/11.4 du
memoire) : cette derniere formule est une APPROXIMATION pedagogique de
l'ordre de grandeur de la composition de la confidentialite. Ce n'est PAS
un comptable de confidentialite (« accountant ») exact. Pour une garantie
publiable, il faut utiliser une bibliotheque dediee comme `opacus`
(PyTorch) ou `dp-accounting` (Google), qui implementent la comptabilite
RDP (Renyi Differential Privacy) exacte. Ce module reste volontairement
simple pour fonctionner sans dependances lourdes ni acces reseau.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict

from model_build import PARAM_KEYS, flatten_dict, unflatten_like


@dataclass
class PrivacyConfig:
    enabled: bool = True
    clip_norm: float = 1.0     # C dans l'Eq. 5.2
    noise_multiplier: float = 1.0  # sigma dans l'Eq. 5.2 (sigma_effectif = noise_multiplier)
    delta: float = 1e-5        # delta cible de la (epsilon,delta)-DP (Def. 5.1)
    sampling_rate: float = 1.0  # q : fraction des exemples locaux utilises par lot


def clip_and_average(per_example_grads: Dict[str, np.ndarray], clip_norm: float) -> Dict[str, np.ndarray]:
    """
    Ecrete chaque gradient INDIVIDUEL a la norme `clip_norm` (Eq. 5.2,
    premiere partie) puis moyenne sur le lot (sans bruit -- le bruit est
    ajoute separement par `add_gaussian_noise`, ce qui permet de tester les
    deux etapes independamment).
    """
    N = per_example_grads["grad_A"].shape[0]

    # Concatene tous les parametres par exemple pour calculer une norme
    # GLOBALE par exemple (et non par parametre), comme l'exige le DP-SGD
    # standard (Abadi et al., 2016).
    flat_per_example = np.concatenate(
        [
            per_example_grads["grad_A"].reshape(N, -1),
            per_example_grads["grad_B"].reshape(N, -1),
            per_example_grads["grad_w_out"].reshape(N, -1),
            per_example_grads["grad_b_out"].reshape(N, -1),
        ],
        axis=1,
    )
    norms = np.linalg.norm(flat_per_example, axis=1)  # (N,)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))  # Eq. 5.2

    clipped_flat = flat_per_example * scale[:, None]
    mean_flat = clipped_flat.mean(axis=0)

    # Redecoupe le vecteur moyen dans la forme (A, B, w_out, b_out).
    template = {
        "A": per_example_grads["grad_A"][0],
        "B": per_example_grads["grad_B"][0],
        "w_out": per_example_grads["grad_w_out"][0],
        "b_out": np.array([per_example_grads["grad_b_out"][0]]),
    }
    sizes = {k: int(np.prod(template[k].shape)) for k in PARAM_KEYS}
    out = {}
    idx = 0
    for k in PARAM_KEYS:
        out[k] = mean_flat[idx: idx + sizes[k]].reshape(template[k].shape)
        idx += sizes[k]
    return out, norms


def add_gaussian_noise(
    mean_grad: Dict[str, np.ndarray], clip_norm: float, noise_multiplier: float,
    batch_size: int, rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Deuxieme partie de l'Eq. 5.2 : ajoute un bruit gaussien d'ecart-type
    (noise_multiplier * clip_norm / batch_size) a la moyenne des gradients
    ecretes.
    """
    noisy = {}
    std = noise_multiplier * clip_norm / max(batch_size, 1)
    for k in PARAM_KEYS:
        noise = rng.normal(0.0, std, size=mean_grad[k].shape)
        noisy[k] = mean_grad[k] + noise
    return noisy


def dp_sgd_step(
    per_example_grads: Dict[str, np.ndarray], cfg: PrivacyConfig, batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Enchaine ecretage + bruitage = un pas complet de DP-SGD (Algorithme 5.1)."""
    mean_grad, norms = clip_and_average(per_example_grads, cfg.clip_norm)
    if not cfg.enabled:
        return mean_grad
    return add_gaussian_noise(mean_grad, cfg.clip_norm, cfg.noise_multiplier, batch_size, rng)


def estimate_epsilon(cfg: PrivacyConfig, steps: int) -> float:
    """
    Estimation APPROXIMATIVE (voir avertissement en tete de fichier) du
    epsilon cumule apres `steps` iterations d'entrainement, suivant la forme
    de l'equation 5.6 du memoire :

        epsilon_total ~= q / sigma * sqrt(T * log(1/delta))

    avec q = cfg.sampling_rate, sigma = cfg.noise_multiplier, T = steps.
    """
    if not cfg.enabled or cfg.noise_multiplier <= 0:
        return float("inf")
    q = cfg.sampling_rate
    sigma = cfg.noise_multiplier
    T = max(steps, 1)
    delta = cfg.delta
    eps = (q / sigma) * np.sqrt(2.0 * T * np.log(1.0 / delta))
    return float(eps)


if __name__ == "__main__":
    from model_build import ModelConfig, BaseModel, LoRAAdapter

    mcfg = ModelConfig()
    base = BaseModel(mcfg)
    rng = np.random.default_rng(1)
    adapter = LoRAAdapter(mcfg, rng)

    X = rng.normal(size=(32, mcfg.d_in))
    y = (rng.uniform(size=32) > 0.7).astype(np.float64)

    grads = adapter.per_example_gradients(base, X, y)
    pcfg = PrivacyConfig(enabled=True, clip_norm=1.0, noise_multiplier=1.0, delta=1e-5, sampling_rate=0.1)
    noisy_grad = dp_sgd_step(grads, pcfg, batch_size=32, rng=rng)

    print("[privacy.py] Formes des gradients bruites :", {k: v.shape for k, v in noisy_grad.items()})
    eps_10 = estimate_epsilon(pcfg, steps=10)
    eps_500 = estimate_epsilon(pcfg, steps=500)
    print(f"[privacy.py] epsilon estime apres 10 pas   = {eps_10:.3f}")
    print(f"[privacy.py] epsilon estime apres 500 pas  = {eps_500:.3f}")
    assert eps_500 > eps_10, "Le budget de confidentialite doit croitre avec le nombre d'iterations."
    print("[privacy.py] auto-test OK.")
