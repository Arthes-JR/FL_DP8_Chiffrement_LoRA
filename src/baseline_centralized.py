"""
baseline_centralized.py
========================
Entraine le MEME modele (chapitre 2.2.4, LoRA) de maniere CENTRALISEE,
c'est-a-dire en regroupant toutes les donnees de tous les clients en un
seul jeu d'entrainement -- exactement la configuration #1 ("Centralise")
du Tableau 11.1 du memoire, servant de BORNE SUPERIEURE de reference a
laquelle comparer les configurations federees (chapitre 11.1.3/11.2.1).

AVERTISSEMENT : ce script n'est utilise QUE pour produire un point de
comparaison scientifique. Il viole delibrement le principe de
non-transfert de donnees brutes (chapitre 3.4.1) et ne doit jamais etre
execute sur des donnees reelles sensibles -- uniquement sur les donnees
SYNTHETIQUES generees par `data_utils.py`.

Usage :
    python baseline_centralized.py --config config/base_config.yaml
"""

from __future__ import annotations
import argparse
import csv
import os
import time

import numpy as np
import yaml

from data_utils import DatasetConfig, generate_synthetic_dataset, train_test_split
from model_build import ModelConfig, BaseModel, LoRAAdapter, PARAM_KEYS
from metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Baseline centralisee (config #1 du memoire).")
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Nombre total d'epoques (par defaut : n_rounds * local_epochs, "
                              "pour un budget de calcul comparable a l'experience federee).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = DatasetConfig(
        task=cfg["dataset"]["task"], n_samples=cfg["dataset"]["n_samples"],
        n_features=cfg["dataset"]["n_features"], test_fraction=cfg["dataset"]["test_fraction"],
        seed=cfg["dataset"]["seed"],
    )
    X, y = generate_synthetic_dataset(dcfg)
    X_train, y_train, X_test, y_test = train_test_split(X, y, dcfg.test_fraction, dcfg.seed)
    print(f"[baseline] {len(X_train)} exemples d'entrainement (CENTRALISES), {len(X_test)} de test.")

    mcfg = ModelConfig(d_in=cfg["dataset"]["n_features"], d_hidden=cfg["model"]["d_hidden"],
                        rank=cfg["model"]["rank"], seed_base=cfg["model"]["seed_base"])
    base = BaseModel(mcfg)
    rng = np.random.default_rng(mcfg.seed_base)
    adapter = LoRAAdapter(mcfg, rng)

    batch_size = cfg["federation"]["local_batch_size"]
    lr = cfg["federation"]["learning_rate"]
    total_epochs = args.epochs or (cfg["federation"]["n_rounds"] * cfg["federation"]["local_epochs"])

    history = []
    t_start = time.time()
    n = len(X_train)
    for epoch in range(1, total_epochs + 1):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            Xb, yb = X_train[idx], y_train[idx]
            grads = adapter.per_example_gradients(base, Xb, yb)
            mean_grads = {
                "grad_A": grads["grad_A"].mean(axis=0), "grad_B": grads["grad_B"].mean(axis=0),
                "grad_w_out": grads["grad_w_out"].mean(axis=0), "grad_b_out": grads["grad_b_out"].mean(),
            }
            adapter.apply_update(mean_grads, lr=lr)

        p_pred = adapter.predict_proba(base, X_test)
        m = compute_metrics(y_test, p_pred)
        history.append({"epoch": epoch, "duration_s": round(time.time() - t_start, 3), **m.as_dict()})
        print(f"[baseline] epoch {epoch:3d}/{total_epochs} | {m}")

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "baseline_centralized_history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    gd = adapter.to_dict()
    np.savez(os.path.join(results_dir, "baseline_centralized_model.npz"), **gd)
    print(f"[baseline] termine en {time.time()-t_start:.1f}s. "
          f"Resultats dans {results_dir}/baseline_centralized_history.csv")


if __name__ == "__main__":
    main()
