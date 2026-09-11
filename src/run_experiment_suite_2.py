"""
run_experiment_suite_2.py
===========================
Deuxieme lot d'experiences reelles, destine A JUSTIFIER les choix
d'hyperparametres retenus par defaut dans base_config.yaml (et non plus
seulement a illustrer le compromis confidentialite/performance).

Produit dans results_suite_2/ :
  n_clients_sweep.csv   -> effet du nombre de clients (scalabilite, chapitre 4.3.4)
  clip_norm_sweep.csv   -> effet du seuil d'ecretage C (Equation 5.2)
  rank_sweep.csv        -> effet du rang LoRA r (Equation 2.2 / 4.4.3), avec
                           nombre de parametres entraines pour chaque r
"""
import copy
import csv
import os
import subprocess
import sys
import time

import yaml

BASE_CFG_PATH = "../config/base_config.yaml"
SUITE_DIR = "../results_suite_2"
os.makedirs(SUITE_DIR, exist_ok=True)

with open(BASE_CFG_PATH) as f:
    BASE = yaml.safe_load(f)


def run_one(cfg, tag, skip_data=False, skip_keys=False):
    cfg_path = f"../config/_tmp2_{tag}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    cmd = [sys.executable, "run_simulation.py", "--config", cfg_path]
    if skip_data:
        cmd.append("--skip-data")
    if skip_keys:
        cmd.append("--skip-keys")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = res.returncode == 0
    print(f"  -> [{tag}] {'OK' if ok else 'ECHEC'} en {dt:.1f}s")
    if not ok:
        print(res.stdout[-2500:]); print(res.stderr[-2500:])
    return ok


def read_metrics(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
# 1) NOMBRE DE CLIENTS : 3, 5, 10, 20 (chapitre 4.3.4, "scalabilite")
# ---------------------------------------------------------------------
print("=== 1) Balayage du nombre de clients ===")
client_counts = [3, 5, 10, 20]
rows_clients = []
for k in client_counts:
    tag = f"nclients_{k}"
    cfg = copy.deepcopy(BASE)
    cfg["federation"]["n_clients"] = k
    cfg["federation"]["n_rounds"] = 12
    cfg["federation"]["aggregation_strategy"] = "secure_fedavg"
    cfg["privacy"]["clip_norm"] = 0.5
    cfg["privacy"]["noise_multiplier"] = 0.3
    cfg["security"]["secure_aggregation"] = True
    cfg["security"]["simulate_dropout"] = None
    # threshold de Shamir doit rester <= k-1
    cfg["security"]["shamir_threshold"] = min(3, max(1, k - 1))
    cfg["paths"]["results_dir"] = f"{SUITE_DIR}/{tag}"
    cfg["paths"]["logs_dir"] = f"{SUITE_DIR}/{tag}_logs"
    t0 = time.time()
    ok = run_one(cfg, tag, skip_data=False, skip_keys=False)
    dt = time.time() - t0
    rows = read_metrics(f"{SUITE_DIR}/{tag}/metrics_history.csv")
    if rows:
        final = rows[-1]
        rows_clients.append({"n_clients": k, "f1_final": final["f1"], "accuracy_final": final["accuracy"],
                              "wall_time_s": round(dt, 1)})

with open(f"{SUITE_DIR}/n_clients_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["n_clients", "f1_final", "accuracy_final", "wall_time_s"])
    w.writeheader(); w.writerows(rows_clients)
print(f"Ecrit {SUITE_DIR}/n_clients_sweep.csv")

# ---------------------------------------------------------------------
# 2) SEUIL D'ECRETAGE C (Equation 5.2)
# ---------------------------------------------------------------------
print("\n=== 2) Balayage du seuil d'ecretage (clip_norm C) ===")
clip_norms = [0.25, 0.5, 1.0, 2.0, 5.0]
rows_clip = []
first = True
for c in clip_norms:
    tag = f"clip_{str(c).replace('.', 'p')}"
    cfg = copy.deepcopy(BASE)
    cfg["federation"]["n_rounds"] = 12
    cfg["federation"]["aggregation_strategy"] = "secure_fedavg"
    cfg["privacy"]["enabled"] = True
    cfg["privacy"]["clip_norm"] = c
    cfg["privacy"]["noise_multiplier"] = 0.6  # valeur mediane du balayage precedent
    cfg["security"]["secure_aggregation"] = True
    cfg["security"]["simulate_dropout"] = None
    cfg["paths"]["results_dir"] = f"{SUITE_DIR}/{tag}"
    cfg["paths"]["logs_dir"] = f"{SUITE_DIR}/{tag}_logs"
    ok = run_one(cfg, tag, skip_data=True, skip_keys=not first)
    first = False
    rows = read_metrics(f"{SUITE_DIR}/{tag}/metrics_history.csv")
    if rows:
        final = rows[-1]
        rows_clip.append({"clip_norm": c, "f1_final": final["f1"], "accuracy_final": final["accuracy"]})

with open(f"{SUITE_DIR}/clip_norm_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["clip_norm", "f1_final", "accuracy_final"])
    w.writeheader(); w.writerows(rows_clip)
print(f"Ecrit {SUITE_DIR}/clip_norm_sweep.csv")

# ---------------------------------------------------------------------
# 3) RANG LoRA r (Equation 2.2 / 4.4.3) : performance vs cout de communication
# ---------------------------------------------------------------------
print("\n=== 3) Balayage du rang LoRA (r) ===")
ranks = [2, 4, 8, 16, 32]
rows_rank = []
d_in = BASE["dataset"]["n_features"]
d_hidden = BASE["model"]["d_hidden"]

first = True
for r in ranks:
    tag = f"rank_{r}"
    cfg = copy.deepcopy(BASE)
    cfg["federation"]["n_rounds"] = 12
    cfg["federation"]["aggregation_strategy"] = "secure_fedavg"
    cfg["privacy"]["enabled"] = True
    cfg["privacy"]["clip_norm"] = 0.5
    cfg["privacy"]["noise_multiplier"] = 0.3
    cfg["security"]["secure_aggregation"] = True
    cfg["security"]["simulate_dropout"] = None
    cfg["model"]["rank"] = r
    cfg["paths"]["results_dir"] = f"{SUITE_DIR}/{tag}"
    cfg["paths"]["logs_dir"] = f"{SUITE_DIR}/{tag}_logs"
    ok = run_one(cfg, tag, skip_data=True, skip_keys=not first)
    first = False
    rows = read_metrics(f"{SUITE_DIR}/{tag}/metrics_history.csv")
    n_params = r * (d_in + d_hidden) + d_hidden + 1  # |A|+|B|+|w_out|+|b_out|
    n_params_full = d_hidden * d_in  # equivalent poids "complets" de la couche adaptee (reference)
    if rows:
        final = rows[-1]
        rows_rank.append({"rank": r, "f1_final": final["f1"], "accuracy_final": final["accuracy"],
                           "n_trainable_params": n_params,
                           "reduction_ratio_pct": round(100 * n_params / n_params_full, 2)})

with open(f"{SUITE_DIR}/rank_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["rank", "f1_final", "accuracy_final", "n_trainable_params", "reduction_ratio_pct"])
    w.writeheader(); w.writerows(rows_rank)
print(f"Ecrit {SUITE_DIR}/rank_sweep.csv")

print("\n=== TERMINE ===")
for name, rows in [("n_clients", rows_clients), ("clip_norm", rows_clip), ("rank", rows_rank)]:
    print(f"-- {name} --")
    for r in rows:
        print(" ", r)
