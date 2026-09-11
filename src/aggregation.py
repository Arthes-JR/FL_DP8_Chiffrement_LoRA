"""
aggregation.py
===============
Regles d'agregation cote serveur.

1) FedAvg pondere (Equation 4.2 du memoire) :
      w_{t+1} = sum_k (n_k/n) * w_{t+1}^k

   Avec agregation securisee (chapitre 5.2), chaque client PRE-MULTIPLIE
   deja sa mise a jour par son poids n_k/n avant masquage (voir client.py),
   de sorte que le serveur n'a plus qu'a SOMMER les contributions demasquees
   pour obtenir directement l'Eq. 4.2 -- c'est la technique standard pour
   combiner FedAvg pondere et agregation securisee (chapitre 9.3.2).

2) Agregation robuste (Equation 4.5) : mediane coordonnee-par-coordonnee et
   moyenne tronquee, utilisees dans les scenarios de stress (chapitre
   8.4.3) en presence de clients malveillants. Ces regles NE SONT PAS
   compatibles avec l'agregation securisee de type Bonawitz (qui ne peut
   calculer qu'une SOMME, pas une mediane) : elles sont donc utilisees en
   alternative, sans masquage, comme discute au chapitre 5.2.4.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List

from model_build import PARAM_KEYS


def fedavg_weighted_sum(updates: List[Dict[str, np.ndarray]], weights: List[float]) -> Dict[str, np.ndarray]:
    """
    Combine des mises a jour DEJA ponderees (chaque `updates[k]` a ete
    multiplie par `weights[k]` en amont, typiquement cote client avant
    masquage) par simple somme -- utilise en mode agregation securisee.
    """
    out = {k: np.zeros_like(updates[0][k]) for k in PARAM_KEYS}
    for upd in updates:
        for k in PARAM_KEYS:
            out[k] = out[k] + upd[k]
    return out


def fedavg_plain(updates: List[Dict[str, np.ndarray]], weights: List[float]) -> Dict[str, np.ndarray]:
    """
    FedAvg standard (Eq. 4.2) SANS agregation securisee : le serveur voit
    chaque mise a jour en clair et applique lui-meme la ponderation n_k/n.
    """
    total_w = sum(weights)
    out = {k: np.zeros_like(updates[0][k]) for k in PARAM_KEYS}
    for upd, w in zip(updates, weights):
        for k in PARAM_KEYS:
            out[k] = out[k] + (w / total_w) * upd[k]
    return out


def robust_median(updates: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Mediane coordonnee-par-coordonnee (premiere partie de l'Eq. 4.5)."""
    out = {}
    for k in PARAM_KEYS:
        stacked = np.stack([u[k] for u in updates], axis=0)  # (K, ...)
        out[k] = np.median(stacked, axis=0)
    return out


def robust_trimmed_mean(updates: List[Dict[str, np.ndarray]], beta: float = 0.2) -> Dict[str, np.ndarray]:
    """
    Moyenne tronquee (seconde partie de l'Eq. 4.5) : pour chaque
    coordonnee, on retire les beta*K valeurs extremes de part et d'autre
    avant de moyenner les valeurs restantes.
    """
    K = len(updates)
    n_trim = int(np.floor(beta * K))
    out = {}
    for k in PARAM_KEYS:
        stacked = np.stack([u[k] for u in updates], axis=0)  # (K, ...)
        sorted_vals = np.sort(stacked, axis=0)
        if n_trim > 0 and K - 2 * n_trim > 0:
            trimmed = sorted_vals[n_trim: K - n_trim]
        else:
            trimmed = sorted_vals
        out[k] = trimmed.mean(axis=0)
    return out


AGGREGATION_STRATEGIES = {
    "fedavg": fedavg_plain,
    "median": robust_median,
    "trimmed_mean": robust_trimmed_mean,
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    template = {"A": rng.normal(size=(4, 5)), "B": rng.normal(size=(6, 4)),
                "w_out": rng.normal(size=(6,)), "b_out": np.array([0.1])}

    updates = []
    for i in range(5):
        updates.append({k: template[k] + rng.normal(scale=0.05, size=template[k].shape) for k in PARAM_KEYS})
    # Un client "malveillant" avec une mise a jour aberrante.
    updates[2] = {k: template[k] * 1000 for k in PARAM_KEYS}

    weights = [10, 20, 5, 15, 30]

    avg = fedavg_plain(updates, weights)
    med = robust_median(updates)
    trimmed = robust_trimmed_mean(updates, beta=0.2)

    print("[aggregation.py] ||FedAvg - vrai||   =", np.linalg.norm(avg["A"] - template["A"]))
    print("[aggregation.py] ||median - vrai||   =", np.linalg.norm(med["A"] - template["A"]))
    print("[aggregation.py] ||trimmed - vrai||  =", np.linalg.norm(trimmed["A"] - template["A"]))

    assert np.linalg.norm(med["A"] - template["A"]) < np.linalg.norm(avg["A"] - template["A"]), \
        "La mediane doit etre plus robuste au client malveillant que FedAvg."
    print("[aggregation.py] auto-test OK : les agregations robustes resistent mieux au client malveillant.")
