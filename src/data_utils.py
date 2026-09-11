"""
data_utils.py
=============
Generation de jeux de donnees SYNTHETIQUES (chapitre 11.1.1 du memoire :
"jeux de donnees synthetiques ou reelles, anonymisees") et partition
non-iid entre clients (chapitre 11.1.1/4.2.2 : "repartition non-iid
controlee" / "gestion de donnees non-iid").

Deux scenarios sont proposes, correspondant aux chapitres 6.2.1/6.2.2 et
8.4.2 du memoire :
  - "health"  : classification binaire d'un risque clinique simule.
  - "finance" : classification binaire d'une transaction frauduleuse simulee.

Aucune donnee reelle n'est utilisee : toutes les caracteristiques et
etiquettes sont generees par un modele statistique connu, ce qui permet de
mesurer l'exactitude du modele federe sans jamais manipuler de donnees
personnelles (conformement a l'exigence 3.4.1 du memoire).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DatasetConfig:
    task: str = "health"          # "health" ou "finance"
    n_samples: int = 4000
    n_features: int = 20
    test_fraction: float = 0.2
    seed: int = 123


def generate_synthetic_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genere un jeu de donnees tabulaire binaire synthetique.

    Le generateur simule une frontiere de decision lineaire bruitee dans un
    espace a `n_features` dimensions, avec un desequilibre de classes
    different selon le scenario (le scenario "finance" simule un fort
    desequilibre, realiste pour la detection de fraude : classe positive
    rare).
    """
    rng = np.random.default_rng(cfg.seed)
    X = rng.normal(0.0, 1.0, size=(cfg.n_samples, cfg.n_features))

    # Vecteur de poids "verite terrain" cache, differe par scenario pour
    # eviter d'utiliser exactement le meme generateur pour les deux taches.
    task_seed_offset = 0 if cfg.task == "health" else 1000
    w_rng = np.random.default_rng(cfg.seed + task_seed_offset)
    true_w = w_rng.normal(0.0, 1.0, size=(cfg.n_features,))
    true_w /= np.linalg.norm(true_w)

    logits = X @ true_w * 2.5

    if cfg.task == "finance":
        # Fort desequilibre de classes (fraude rare) : on decale le biais
        # pour obtenir environ 5-8% de positifs.
        bias = np.quantile(logits, 0.93)
        logits = logits - bias
    else:
        # Scenario sante : desequilibre plus modere (~30% de positifs).
        bias = np.quantile(logits, 0.70)
        logits = logits - bias

    noise = rng.normal(0.0, 0.6, size=cfg.n_samples)
    p = 1.0 / (1.0 + np.exp(-(logits + noise)))
    y = (rng.uniform(size=cfg.n_samples) < p).astype(np.float64)

    return X, y


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_fraction)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def partition_non_iid_dirichlet(
    X: np.ndarray, y: np.ndarray, n_clients: int, alpha: float, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partitionne (X, y) entre n_clients de maniere non-iid, en utilisant une
    distribution de Dirichlet sur les proportions de classes par client
    (technique standard, cf. Hsu et al., 2019 ; correspond au chapitre
    11.1.1/11.1.2 du memoire : "repartition non-iid controlee").

    alpha petit (< 1)  => forte heterogeneite (chaque client voit surtout
                           une seule classe, cas extreme non-iid).
    alpha grand (> 10) => repartition proche de l'iid (chaque client voit
                           un melange equilibre des classes).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        # Proportions de la classe c allouees a chaque client (Dirichlet).
        proportions = rng.dirichlet(alpha=np.full(n_clients, alpha))
        # Convertit les proportions en indices de decoupe cumulatifs.
        cut_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, cut_points)
        for k in range(n_clients):
            client_indices[k].extend(splits[k].tolist())

    shards = []
    for k in range(n_clients):
        idx = np.array(client_indices[k], dtype=int)
        rng.shuffle(idx)
        shards.append((X[idx], y[idx]))
    return shards


def describe_partition(shards: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    lines = ["Repartition non-iid des donnees entre clients :"]
    for i, (Xk, yk) in enumerate(shards):
        n = len(yk)
        pos = int(yk.sum())
        lines.append(
            f"  - Client {i+1}: {n:5d} exemples  |  positifs={pos:4d} ({100*pos/max(n,1):5.1f}%)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = DatasetConfig(task="finance", n_samples=4000, n_features=20, seed=123)
    X, y = generate_synthetic_dataset(cfg)
    print(f"[data_utils.py] X.shape={X.shape}  taux de positifs global={y.mean():.3f}")

    X_train, y_train, X_test, y_test = train_test_split(X, y, cfg.test_fraction, cfg.seed)
    print(f"[data_utils.py] train={len(X_train)}  test={len(X_test)}")

    shards = partition_non_iid_dirichlet(X_train, y_train, n_clients=5, alpha=0.5, seed=cfg.seed)
    print(describe_partition(shards))
    print("[data_utils.py] auto-test OK.")
