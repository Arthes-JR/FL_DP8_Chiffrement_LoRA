"""
run_experiment_suite.py
========================
Execute PLUSIEURS experiences reelles (pas une seule config), pour produire
de vraies courbes de convergence et un vrai compromis confidentialite/
performance, destines au chapitre 11 du memoire et aux slides.

Produit :
  results_suite/curves_by_config.csv   -> F1 par round, pour les 4 configurations du Tableau 11.1
  results_suite/privacy_sweep.csv      -> F1 final et epsilon estime, pour plusieurs sigma
  results_suite/alpha_sweep.csv        -> F1 final, pour plusieurs niveaux de non-iid (alpha)
  results_suite/summary.json           -> recapitulatif
"""
import copy
import csv
import json
import os
import subprocess
import sys
import time

import yaml

BASE_CFG_PATH = "../config/base_config.yaml"
SUITE_DIR = "../results_suite"
os.makedirs(SUITE_DIR, exist_ok=True)

with open(BASE_CFG_PATH) as f:
    BASE = yaml.safe_load(f)


def run_one(cfg, tag, skip_data=False, skip_keys=False):
    cfg_path = f"../config/_tmp_{tag}.yaml"
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
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
    return ok


def read_metrics(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
# 1) COURBES DE CONVERGENCE : les 4 configurations du Tableau 11.1
# ---------------------------------------------------------------------
print("=== 1) Courbes de convergence (4 configurations) ===")

configs_for_curves = {
    "fl_no_dp": {"aggregation_strategy": "plain_fedavg", "privacy_enabled": False, "secure_agg": False},
    "fl_dp_no_secagg": {"aggregation_strategy": "plain_fedavg", "privacy_enabled": True, "secure_agg": False},
    "fl_dp_secagg": {"aggregation_strategy": "secure_fedavg", "privacy_enabled": True, "secure_agg": True},
    "fl_dp_median_stress": {"aggregation_strategy": "plain_median", "privacy_enabled": True, "secure_agg": False},
}

all_curves = []  # rows: config, round, accuracy, precision, recall, f1, loss

first = True
for tag, opts in configs_for_curves.items():
    cfg = copy.deepcopy(BASE)
    cfg["federation"]["aggregation_strategy"] = opts["aggregation_strategy"]
    cfg["privacy"]["enabled"] = opts["privacy_enabled"]
    cfg["security"]["secure_aggregation"] = opts["secure_agg"]
    cfg["security"]["simulate_dropout"] = None
    cfg["paths"]["results_dir"] = f"{SUITE_DIR}/{tag}"
    cfg["paths"]["logs_dir"] = f"{SUITE_DIR}/{tag}_logs"
    ok = run_one(cfg, tag, skip_data=not first, skip_keys=not first)
    first = False
    rows = read_metrics(f"{SUITE_DIR}/{tag}/metrics_history.csv")
    for r in rows:
        all_curves.append({"config": tag, "round": r["round"], "f1": r["f1"],
                            "accuracy": r["accuracy"], "loss": r["loss"]})

# Baseline centralisee (rejouee une seule fois, meme donnees).
print("  -> [baseline_centralized] en cours...")
base_cfg = copy.deepcopy(BASE)
base_cfg["paths"]["results_dir"] = f"{SUITE_DIR}/baseline"
with open(f"../config/_tmp_baseline.yaml", "w") as f:
    yaml.dump(base_cfg, f, allow_unicode=True)
res = subprocess.run([sys.executable, "baseline_centralized.py", "--config", "../config/_tmp_baseline.yaml",
                       "--epochs", str(BASE["federation"]["n_rounds"] * BASE["federation"]["local_epochs"])],
                      capture_output=True, text=True)
print("  -> [baseline_centralized]", "OK" if res.returncode == 0 else "ECHEC")
if res.returncode != 0:
    print(res.stdout[-2000:], res.stderr[-2000:])
base_rows = read_metrics(f"{SUITE_DIR}/baseline/baseline_centralized_history.csv")
# On ne garde qu'un point tous les `local_epochs` epoques pour aligner sur les "rounds" federes.
le = BASE["federation"]["local_epochs"]
for r in base_rows:
    if int(r["epoch"]) % le == 0:
        rnd = int(r["epoch"]) // le
        all_curves.append({"config": "centralized_baseline", "round": rnd, "f1": r["f1"],
                            "accuracy": r["accuracy"], "loss": r["loss"]})

with open(f"{SUITE_DIR}/curves_by_config.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "round", "f1", "accuracy", "loss"])
    w.writeheader()
    w.writerows(all_curves)
print(f"Ecrit {SUITE_DIR}/curves_by_config.csv ({len(all_curves)} lignes)")

# ---------------------------------------------------------------------
# 2) BALAYAGE DE CONFIDENTIALITE : plusieurs sigma (noise_multiplier)
# ---------------------------------------------------------------------
print("\n=== 2) Balayage du bruit differentiel (sigma) ===")
sigmas = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
privacy_rows = []

from privacy import PrivacyConfig, estimate_epsilon  # noqa

first = True
for sigma in sigmas:
    tag = f"sigma_{str(sigma).replace('.', 'p')}"
    cfg = copy.deepcopy(BASE)
    cfg["federation"]["aggregation_strategy"] = "secure_fedavg"
    cfg["privacy"]["enabled"] = True
    cfg["privacy"]["noise_multiplier"] = sigma
    cfg["federation"]["n_rounds"] = 12
    cfg["security"]["secure_aggregation"] = True
    cfg["security"]["simulate_dropout"] = None
    cfg["paths"]["results_dir"] = f"{SUITE_DIR}/{tag}"
    cfg["paths"]["logs_dir"] = f"{SUITE_DIR}/{tag}_logs"
    ok = run_one(cfg, tag, skip_data=True, skip_keys=not first)
    first = False
    rows = read_metrics(f"{SUITE_DIR}/{tag}/metrics_history.csv")
    if rows:
        final = rows[-1]
        n_rounds = BASE["federation"]["n_rounds"]
        local_epochs = BASE["federation"]["local_epochs"]
        batches_per_epoch = max(1, (BASE["dataset"]["n_samples"] // BASE["federation"]["n_clients"]) // BASE["federation"]["local_batch_size"])
        total_steps = n_rounds * local_epochs * batches_per_epoch
        pcfg = PrivacyConfig(enabled=True, clip_norm=BASE["privacy"]["clip_norm"], noise_multiplier=sigma,
                              delta=BASE["privacy"]["delta"], sampling_rate=BASE["privacy"]["sampling_rate"])
        eps = estimate_epsilon(pcfg, steps=total_steps)
        privacy_rows.append({"sigma": sigma, "epsilon_estime": round(eps, 3), "f1_final": final["f1"],
                              "accuracy_final": final["accuracy"]})

with open(f"{SUITE_DIR}/privacy_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sigma", "epsilon_estime", "f1_final", "accuracy_final"])
    w.writeheader()
    w.writerows(privacy_rows)
print(f"Ecrit {SUITE_DIR}/privacy_sweep.csv ({len(privacy_rows)} lignes)")

# ---------------------------------------------------------------------
# 3) BALAYAGE NON-IID : plusieurs alpha (Dirichlet)
# ---------------------------------------------------------------------
print("\n=== 3) Balayage de l'heterogeneite non-iid (alpha) ===")
alphas = [0.1, 0.3, 0.5, 1.0, 5.0]
alpha_rows = []

for alpha in alphas:
    tag = f"alpha_{str(alpha).replace('.', 'p')}"
    cfg = copy.deepcopy(BASE)
    cfg["federation"]["aggregation_strategy"] = "secure_fedavg"
    cfg["federation"]["non_iid_alpha"] = alpha
    cfg["federation"]["n_rounds"] = 12
    cfg["privacy"]["enabled"] = True
    cfg["security"]["secure_aggregation"] = True
    cfg["security"]["simulate_dropout"] = None
    cfg["paths"]["results_dir"] = f"{SUITE_DIR}/{tag}"
    cfg["paths"]["logs_dir"] = f"{SUITE_DIR}/{tag}_logs"
    ok = run_one(cfg, tag, skip_data=False, skip_keys=False)  # alpha change la partition -> regenerer les donnees
    rows = read_metrics(f"{SUITE_DIR}/{tag}/metrics_history.csv")
    if rows:
        final = rows[-1]
        alpha_rows.append({"alpha": alpha, "f1_final": final["f1"], "accuracy_final": final["accuracy"]})

with open(f"{SUITE_DIR}/alpha_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["alpha", "f1_final", "accuracy_final"])
    w.writeheader()
    w.writerows(alpha_rows)
print(f"Ecrit {SUITE_DIR}/alpha_sweep.csv ({len(alpha_rows)} lignes)")

# ---------------------------------------------------------------------
# Recapitulatif
# ---------------------------------------------------------------------
summary = {
    "curves_final_f1": {
        tag: [r["f1"] for r in all_curves if r["config"] == tag][-1] if any(r["config"] == tag for r in all_curves) else None
        for tag in list(configs_for_curves.keys()) + ["centralized_baseline"]
    },
    "privacy_sweep": privacy_rows,
    "alpha_sweep": alpha_rows,
}
with open(f"{SUITE_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\n=== TERMINE ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))
