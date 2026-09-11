"""
visualize.py
============
Genere TOUTES les figures de resultats du memoire (chapitre 11 et annexe)
a partir des fichiers CSV produits par run_simulation.py,
run_experiment_suite.py et run_experiment_suite_2.py.

Ce module est le pendant Python des figures pgfplots integrees dans le
rapport LaTeX : il permet de regenerer les memes graphiques (au format
PNG haute resolution) directement a partir de vos propres resultats,
sans avoir a reecrire de code LaTeX.

Usage :
    python visualize.py --suite1 ../results_suite --suite2 ../results_suite_2 --out ../figures

Produit dans le dossier de sortie :
    fig_convergence_curves.png   -- F1 par round, 5 configurations (chapitre 11.2.1)
    fig_privacy_sweep.png        -- F1 final vs sigma (chapitre 11.2.2)
    fig_alpha_sweep.png          -- F1 final vs alpha non-iid (chapitre 11.2.4)
    fig_n_clients_sweep.png      -- F1 final vs nombre de clients (annexe)
    fig_clip_norm_sweep.png      -- F1 final vs seuil d'ecretage C (annexe)
    fig_rank_sweep.png           -- F1 final ET cout de communication vs rang LoRA (annexe)
    fig_summary_bar.png          -- comparaison finale des 4 configurations (barres)
"""

from __future__ import annotations
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette coherente avec le rapport LaTeX (bleu "tfcblue" + accents).
NAVY = "#14325A"
BLUE = "#3D6FB4"
GREEN = "#1F9D6B"
RED = "#C0392B"
AMBER = "#E8963C"
GREY = "#6B7280"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#E5E9F2",
    "grid.linewidth": 0.8,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def read_csv(path):
    if not os.path.exists(path):
        print(f"  [!] fichier introuvable, ignore : {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_convergence_curves(suite1_dir, out_dir):
    rows = read_csv(os.path.join(suite1_dir, "curves_by_config.csv"))
    if not rows:
        return
    configs = {
        "centralized_baseline": ("Centralisé", NAVY, "-"),
        "fl_no_dp": ("Fédéré, sans DP", BLUE, "-"),
        "fl_dp_secagg": ("Fédéré + DP + agrég. sécurisée", GREEN, "-"),
        "fl_dp_median_stress": ("Fédéré + DP + médiane (stress)", RED, "--"),
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag, (label, color, style) in configs.items():
        sub = [r for r in rows if r["config"] == tag]
        if not sub:
            continue
        x = [int(r["round"]) for r in sub]
        y = [float(r["f1"]) for r in sub]
        ax.plot(x, y, style, color=color, linewidth=2, label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Score F1 (jeu de test tenu par le serveur)")
    ax.set_title("Courbes de convergence réelles par configuration")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 0.62)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_convergence_curves.png"), dpi=180)
    plt.close(fig)
    print("  -> fig_convergence_curves.png")


def fig_summary_bar(suite1_dir, out_dir):
    rows = read_csv(os.path.join(suite1_dir, "curves_by_config.csv"))
    if not rows:
        return
    order = [
        ("centralized_baseline", "Centralisé"), ("fl_no_dp", "Fédéré\nsans DP"),
        ("fl_dp_secagg", "Fédéré + DP +\nagrég. sécurisée"), ("fl_dp_median_stress", "Fédéré + DP +\nmédiane (stress)"),
    ]
    values = []
    for tag, _ in order:
        sub = [r for r in rows if r["config"] == tag]
        values.append(float(sub[-1]["f1"]) if sub else 0.0)
    labels = [lbl for _, lbl in order]
    colors = [NAVY, BLUE, GREEN, RED]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Score F1 final (dernier round)")
    ax.set_title("Comparaison finale des 4 configurations")
    ax.set_ylim(0, max(values) * 1.25 + 0.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_summary_bar.png"), dpi=180)
    plt.close(fig)
    print("  -> fig_summary_bar.png")


def fig_privacy_sweep(suite1_dir, out_dir):
    rows = read_csv(os.path.join(suite1_dir, "privacy_sweep.csv"))
    if not rows:
        return
    x = [float(r["sigma"]) for r in rows]
    y = [float(r["f1_final"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, y, "-o", color=NAVY, linewidth=2, markersize=6)
    ax.set_xlabel("Écart-type du bruit gaussien σ (Équation 5.2)")
    ax.set_ylabel("Score F1 final")
    ax.set_title("Compromis confidentialité / performance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_privacy_sweep.png"), dpi=180)
    plt.close(fig)
    print("  -> fig_privacy_sweep.png")


def fig_alpha_sweep(suite1_dir, out_dir):
    rows = read_csv(os.path.join(suite1_dir, "alpha_sweep.csv"))
    if not rows:
        return
    x = [float(r["alpha"]) for r in rows]
    y = [float(r["f1_final"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, y, "-o", color=AMBER, linewidth=2, markersize=6)
    ax.set_xlabel("Paramètre de Dirichlet α (petit = très non-iid)")
    ax.set_ylabel("Score F1 final")
    ax.set_title("Sensibilité à l'hétérogénéité non-iid")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_alpha_sweep.png"), dpi=180)
    plt.close(fig)
    print("  -> fig_alpha_sweep.png")


def fig_n_clients_sweep(suite2_dir, out_dir):
    rows = read_csv(os.path.join(suite2_dir, "n_clients_sweep.csv"))
    if not rows:
        return
    x = [int(r["n_clients"]) for r in rows]
    y = [float(r["f1_final"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, y, "-o", color=GREEN, linewidth=2, markersize=6)
    ax.set_xlabel("Nombre de clients (taille totale des données fixée)")
    ax.set_ylabel("Score F1 final")
    ax.set_title("Effet du nombre de clients (scalabilité)")
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_n_clients_sweep.png"), dpi=180)
    plt.close(fig)
    print("  -> fig_n_clients_sweep.png")


def fig_clip_norm_sweep(suite2_dir, out_dir):
    rows = read_csv(os.path.join(suite2_dir, "clip_norm_sweep.csv"))
    if not rows:
        return
    x = [float(r["clip_norm"]) for r in rows]
    y = [float(r["f1_final"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, y, "-o", color=RED, linewidth=2, markersize=6)
    ax.set_xlabel("Seuil d'écrêtage C (Équation 5.2)")
    ax.set_ylabel("Score F1 final")
    ax.set_title("Effet du seuil d'écrêtage des gradients")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_clip_norm_sweep.png"), dpi=180)
    plt.close(fig)
    print("  -> fig_clip_norm_sweep.png")


def fig_rank_sweep(suite2_dir, out_dir):
    rows = read_csv(os.path.join(suite2_dir, "rank_sweep.csv"))
    if not rows:
        return
    x = [int(r["rank"]) for r in rows]
    f1 = [float(r["f1_final"]) for r in rows]
    comm = [float(r["reduction_ratio_pct"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5.2))
    ax1.plot(x, f1, "-o", color=NAVY, linewidth=2, markersize=6, label="Score F1 final")
    ax1.set_xlabel("Rang LoRA r")
    ax1.set_ylabel("Score F1 final", color=NAVY)
    ax1.tick_params(axis="y", labelcolor=NAVY)

    ax2 = ax1.twinx()
    ax2.plot(x, comm, "--s", color=AMBER, linewidth=2, markersize=6, label="Coût relatif (% de la couche complète)")
    ax2.set_ylabel("Paramètres LoRA / paramètres complets (%)", color=AMBER)
    ax2.tick_params(axis="y", labelcolor=AMBER)
    ax2.grid(False)
    ax2.axhline(100, color=AMBER, linestyle=":", linewidth=1)

    ax1.set_title("Compromis performance / coût de communication selon le rang LoRA")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_rank_sweep.png"), dpi=180)
    plt.close(fig)
    print("  -> fig_rank_sweep.png")


def main():
    parser = argparse.ArgumentParser(description="Genere les figures de resultats du memoire (chapitre 11 + annexe).")
    parser.add_argument("--suite1", default="../results_suite")
    parser.add_argument("--suite2", default="../results_suite_2")
    parser.add_argument("--out", default="../figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Génération des figures dans {args.out}/ ...")

    fig_convergence_curves(args.suite1, args.out)
    fig_summary_bar(args.suite1, args.out)
    fig_privacy_sweep(args.suite1, args.out)
    fig_alpha_sweep(args.suite1, args.out)
    fig_n_clients_sweep(args.suite2, args.out)
    fig_clip_norm_sweep(args.suite2, args.out)
    fig_rank_sweep(args.suite2, args.out)

    print("Terminé.")


if __name__ == "__main__":
    main()
