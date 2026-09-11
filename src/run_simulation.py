"""
run_simulation.py
==================
Orchestrateur de bout en bout pour lancer une simulation LOCALE complete du
prototype decrit aux chapitres 9 et 10 du memoire, sur une seule machine :

  1) Genere le jeu de donnees synthetique et sa partition non-iid entre
     clients (chapitre 11.1.1), et sauvegarde chaque shard sur disque.
  2) Genere les cles et certificats (`key_generate.py`).
  3) Lance `server.py` et un `client.py` PAR CLIENT comme de VRAIS PROCESSUS
     SYSTEME independants (subprocess), communiquant par sockets TCP+TLS
     sur `127.0.0.1` -- exactement comme ils communiqueraient entre
     machines distinctes sur un reseau reel (chapitre 9.4, chapitre 10).
  4) Attend la fin de l'experience puis affiche un resume des resultats
     (dernieres metriques, evolution round par round).

Ce script est le point d'entree recommande pour reproduire les resultats
du chapitre 11 du memoire. Pour un deploiement reellement distribue
(chaque client sur sa propre machine), lancer directement `server.py` sur
la machine serveur et `client.py --id k` sur chaque machine cliente, en
adaptant `network.host`/`network.port` dans `config/base_config.yaml`
(voir README.md).

Usage :
    python run_simulation.py --config config/base_config.yaml
"""

from __future__ import annotations
import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
import yaml

import key_generate
from data_utils import (
    DatasetConfig, generate_synthetic_dataset, train_test_split,
    partition_non_iid_dirichlet, describe_partition,
)


def prepare_data(cfg: dict) -> None:
    dcfg = DatasetConfig(
        task=cfg["dataset"]["task"], n_samples=cfg["dataset"]["n_samples"],
        n_features=cfg["dataset"]["n_features"], test_fraction=cfg["dataset"]["test_fraction"],
        seed=cfg["dataset"]["seed"],
    )
    X, y = generate_synthetic_dataset(dcfg)
    X_train, y_train, X_test, y_test = train_test_split(X, y, dcfg.test_fraction, dcfg.seed)

    data_dir = cfg["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    np.savez(os.path.join(data_dir, "test_set.npz"), X=X_test, y=y_test)

    shards = partition_non_iid_dirichlet(
        X_train, y_train, n_clients=cfg["federation"]["n_clients"],
        alpha=cfg["federation"]["non_iid_alpha"], seed=dcfg.seed,
    )
    print(describe_partition(shards))
    for i, (Xk, yk) in enumerate(shards, start=1):
        np.savez(os.path.join(data_dir, f"client_{i}.npz"), X=Xk, y=yk)
    print(f"[run_simulation] donnees generees et partitionnees dans {data_dir}/")


def prepare_keys(cfg: dict) -> None:
    key_generate.generate_all_keys(
        n_clients=cfg["federation"]["n_clients"],
        threshold=cfg["security"]["shamir_threshold"],
        out_dir=cfg["paths"]["keys_dir"],
    )


def launch_processes(config_path: str, n_clients: int, logs_dir: str):
    os.makedirs(logs_dir, exist_ok=True)
    procs = []

    server_log = open(os.path.join(logs_dir, "server.log"), "w")
    server_proc = subprocess.Popen(
        [sys.executable, "server.py", "--config", config_path],
        stdout=server_log, stderr=subprocess.STDOUT,
    )
    procs.append(("server", server_proc, server_log))
    time.sleep(1.5)  # laisse au serveur le temps d'ouvrir le port d'ecoute

    for cid in range(1, n_clients + 1):
        client_log = open(os.path.join(logs_dir, f"client_{cid}.log"), "w")
        client_proc = subprocess.Popen(
            [sys.executable, "client.py", "--id", str(cid), "--config", config_path],
            stdout=client_log, stderr=subprocess.STDOUT,
        )
        procs.append((f"client_{cid}", client_proc, client_log))
        time.sleep(0.2)

    return procs


def wait_and_report(procs) -> int:
    exit_codes = {}
    for name, proc, logf in procs:
        code = proc.wait()
        logf.close()
        exit_codes[name] = code
        status = "OK" if code == 0 else f"ECHEC (code {code})"
        print(f"[run_simulation] processus '{name}' termine : {status}")
    return max(exit_codes.values())


def print_summary(results_dir: str) -> None:
    csv_path = os.path.join(results_dir, "metrics_history.csv")
    if not os.path.exists(csv_path):
        print("[run_simulation] AUCUN historique de metriques trouve -- l'experience a-t-elle echoue ?")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[run_simulation] historique de metriques vide.")
        return

    print("\n" + "=" * 78)
    print("RESUME DE L'EXPERIENCE (voir aussi results/metrics_history.csv)")
    print("=" * 78)
    header = f"{'round':>6} | {'actifs':>6} | {'acc':>6} | {'prec':>6} | {'rappel':>6} | {'F1':>6} | {'perte':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['round']:>6} | {r['n_active_clients']:>6} | {r['accuracy']:>6} | "
              f"{r['precision']:>6} | {r['recall']:>6} | {r['f1']:>6} | {r['loss']:>7}")
    last = rows[-1]
    print("-" * len(header))
    print(f"Resultat final (round {last['round']}) : "
          f"accuracy={last['accuracy']}  F1={last['f1']}  perte={last['loss']}")
    print(f"Modele final : {os.path.join(results_dir, 'global_model_final.npz')}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Orchestrateur de simulation federee locale.")
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--skip-data", action="store_true", help="Ne pas regenerer les donnees.")
    parser.add_argument("--skip-keys", action="store_true", help="Ne pas regenerer les cles.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"[run_simulation] experience : {cfg['experiment_name']}")
    print(f"[run_simulation] strategie d'agregation : {cfg['federation']['aggregation_strategy']}")

    if not args.skip_data:
        prepare_data(cfg)
    if not args.skip_keys:
        prepare_keys(cfg)

    t0 = time.time()
    procs = launch_processes(args.config, cfg["federation"]["n_clients"], cfg["paths"]["logs_dir"])
    worst_code = wait_and_report(procs)
    duration = time.time() - t0
    print(f"[run_simulation] experience terminee en {duration:.1f}s (code de sortie max = {worst_code}).")

    print_summary(cfg["paths"]["results_dir"])


if __name__ == "__main__":
    main()
