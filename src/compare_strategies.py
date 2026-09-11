"""
compare_strategies.py
=====================
Lance 6 experiences federees (3 strategies x 2 scenarios) et trace les
courbes F1 superposees.

Configs utilisees (celles qui existent dans config/) :
  - base_config.yaml          -> secure_fedavg  (masquage + Shamir)
  - config_median.yaml        -> plain_median   (agregation robuste mediane)
  - config_trimmed.yaml       -> plain_trimmed  (moyenne tronquee beta=0.2)

Scenarios :
  - sans decrochage
  - avec decrochage (client 3 au round 5)

Sortie :
  - figures/compare_strategies.png  (courbes superposees + barres)
  - figures/compare_strategies.csv  (tableau recapitulatif)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CONFIG_DIR = ROOT / "config"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Chemins vers TES configs existantes
# --------------------------------------------------------------------------
CFG_SECURE_FEDAVG = CONFIG_DIR / "base_config.yaml"
CFG_MEDIAN        = CONFIG_DIR / "config_median.yaml"
CFG_TRIMMED       = CONFIG_DIR / "config_trimmed.yaml"

# --------------------------------------------------------------------------
# Definition des runs a executer
# --------------------------------------------------------------------------
RUNS = [
    # label,                                 config_path,        dropout, suffix
    ("A : FedAvg secure (sans decrochage)",  CFG_SECURE_FEDAVG,  False,   "A"),
    ("B : FedAvg secure (avec decrochage)",  CFG_SECURE_FEDAVG,  True,    "B"),
    ("C : Mediane       (sans decrochage)",  CFG_MEDIAN,         False,   "C"),
    ("D : Mediane       (avec decrochage)",  CFG_MEDIAN,         True,    "D"),
    ("E : Trimmed       (sans decrochage)",  CFG_TRIMMED,        False,   "E"),
    ("F : Trimmed       (avec decrochage)",  CFG_TRIMMED,        True,    "F"),
]


def make_temp_config(base_cfg: Path, dropout: bool, suffix: str) -> Path:
    """Cree une config temporaire derivee d'une config existante."""
    with open(base_cfg, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("security", {})
    if dropout:
        cfg["security"]["simulate_dropout"] = {"client_id": 3, "round": 5}
    else:
        cfg["security"]["simulate_dropout"] = None

    cfg.setdefault("paths", {})
    cfg["paths"]["results_dir"] = f"results_{suffix}"
    cfg["paths"]["logs_dir"] = f"logs_{suffix}"

    out = CONFIG_DIR / f"tmp_{suffix}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def clean_outputs(suffix: str) -> None:
    for d in (SRC / f"results_{suffix}", SRC / f"logs_{suffix}"):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def run_one(label: str, base_cfg: Path, dropout: bool, suffix: str) -> dict:
    print(f"\n{'='*78}")
    print(f"  RUN {suffix}  --  {label}")
    print(f"  config = {base_cfg.name}   dropout = {dropout}")
    print(f"{'='*78}")

    if not base_cfg.exists():
        print(f"  [!] config introuvable : {base_cfg}")
        return {"label": label, "config": base_cfg.name, "dropout": dropout,
                "duration": 0.0, "ok": False, "df": None}

    tmp_cfg = make_temp_config(base_cfg, dropout, suffix)
    clean_outputs(suffix)

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "run_simulation.py", "--config", str(tmp_cfg)],
        cwd=str(SRC),
        capture_output=True,
        text=True,
    )
    duration = time.time() - t0

    if proc.stdout:
        for line in proc.stdout.splitlines()[-25:]:
            print("   |", line)

    csv_path = SRC / f"results_{suffix}" / "metrics_history.csv"
    if proc.returncode != 0 or not csv_path.exists():
        print(f"  [!] ECHEC du run {suffix} (code={proc.returncode})")
        if proc.stderr:
            print(proc.stderr[-1200:])
        return {"label": label, "config": base_cfg.name, "dropout": dropout,
                "duration": duration, "ok": False, "df": None}

    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    eps = float(last["epsilon"]) if "epsilon" in df.columns else float("nan")
    print(f"  -> OK  F1={last['f1']:.4f}  acc={last['accuracy']:.4f}  "
          f"eps={eps:.3f}  ({duration:.1f}s)")

    return {"label": label, "config": base_cfg.name, "dropout": dropout,
            "duration": duration, "ok": True, "df": df}


def plot_curves(results: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Axe 1 : F1 par round, tous les runs
    ax = axes[0]
    for r in results:
        if not r["ok"] or r["df"] is None:
            continue
        style = "-" if r["dropout"] else "--"
        ax.plot(r["df"]["round"], r["df"]["f1"], style,
                label=r["label"], linewidth=1.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("F1")
    ax.set_title("Evolution du F1 -- 6 strategies federees")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # --- Axe 2 : barres du F1 final
    ax = axes[1]
    labels, f1s, colors = [], [], []
    palette = ["#2E86DE", "#54A0FF", "#EE5A24", "#F79F1F", "#A29BFE", "#6C5CE7"]
    i = 0
    for r in results:
        if r["ok"] and r["df"] is not None:
            labels.append(r["label"])
            f1s.append(r["df"]["f1"].iloc[-1])
            colors.append(palette[i % len(palette)])
            i += 1
    bars = ax.bar(range(len(labels)), f1s, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("F1 final")
    ax.set_title("F1 final par strategie")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "compare_strategies.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[compare_strategies] figure sauvegardee : {out}")


def write_summary_table(results: list) -> None:
    rows = []
    for r in results:
        if not r["ok"] or r["df"] is None:
            continue
        last = r["df"].iloc[-1]
        eps = float(last["epsilon"]) if "epsilon" in r["df"].columns else float("nan")
        rows.append({
            "run": r["label"],
            "config": r["config"],
            "decrochage": "oui" if r["dropout"] else "non",
            "rounds": int(last["round"]),
            "accuracy": round(float(last["accuracy"]), 4),
            "precision": round(float(last["precision"]), 4),
            "rappel": round(float(last["recall"]), 4),
            "F1": round(float(last["f1"]), 4),
            "perte": round(float(last["loss"]), 4),
            "epsilon": round(eps, 3) if eps == eps else "",
            "duree_s": round(r["duration"], 1),
        })
    df = pd.DataFrame(rows)
    out = FIG_DIR / "compare_strategies.csv"
    df.to_csv(out, index=False)
    print(f"[compare_strategies] tableau sauvegarde : {out}")
    print("\n=== TABLEAU RECAPITULATIF ===")
    print(df.to_string(index=False))


def main():
    print(f"[compare_strategies] ROOT = {ROOT}")
    print(f"[compare_strategies] SRC  = {SRC}")
    print(f"[compare_strategies] configs disponibles dans {CONFIG_DIR} :")
    for f in sorted(CONFIG_DIR.glob("*.yaml")):
        print(f"   - {f.name}")

    results = []
    for label, cfg_path, dropout, suffix in RUNS:
        r = run_one(label, cfg_path, dropout, suffix)
        results.append(r)

    # Nettoyage des configs temporaires
    for cfg in CONFIG_DIR.glob("tmp_*.yaml"):
        try:
            cfg.unlink()
        except Exception:
            pass

    plot_curves(results)
    write_summary_table(results)
    print("\n[compare_strategies] TERMINE.")


if __name__ == "__main__":
    main()