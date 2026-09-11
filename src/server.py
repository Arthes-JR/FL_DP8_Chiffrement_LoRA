"""
server.py
=========
Serveur federateur (chapitre 9.3 du memoire). Implemente :

  - la phase de configuration ("setup", chapitre 9.3.1) : accepte les
    connexions TLS des K clients, recoit leur identifiant et la taille de
    leur jeu de donnees local (n_k), calcule les poids FedAvg (n_k/n) ;

  - la boucle de rounds (Algorithme 4.1, FedAvg) : diffuse le modele
    global courant, recoit les mises a jour (en clair ou masquees selon la
    strategie), detecte les decrochages, demande aux clients survivants de
    reveler leurs parts de Shamir si necessaire (Algorithme 5.2), agrege
    (FedAvg pondere ou agregation robuste), evalue sur un jeu de test
    tenu par le serveur (a des fins de demonstration uniquement -- ce jeu
    de test n'est JAMAIS utilise pour l'entrainement) ;

  - la sauvegarde du modele final "pret pour la production" (chapitre
    9.3.3) et de l'historique des metriques (chapitre 11, resultats).

Ce fichier n'implemente PAS de parallelisme reseau (un seul thread,
communications sequentielles avec chaque client a chaque round) : ce choix
de simplicite est documente et discute au chapitre 4.3.4 du memoire
("scalabilite a grande echelle") comme une limite assumee du prototype.

Usage :
    python server.py --config config/base_config.yaml
"""

from __future__ import annotations
import argparse
import base64
import csv
import json
import os
import socket
import ssl
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import crypto
import net_protocol as net
from model_build import ModelConfig, BaseModel, LoRAAdapter, PARAM_KEYS
from aggregation import fedavg_weighted_sum, fedavg_plain, robust_median, robust_trimmed_mean
from metrics import compute_metrics
from privacy import estimate_epsilon, PrivacyConfig


def log(msg: str) -> None:
    print(f"[server] {msg}", flush=True)


class FederatedServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.host = cfg["network"]["host"]
        self.port = cfg["network"]["port"]
        self.n_clients = cfg["federation"]["n_clients"]
        self.n_rounds = cfg["federation"]["n_rounds"]
        self.strategy = cfg["federation"]["aggregation_strategy"]
        self.round_timeout = cfg["network"]["round_timeout_s"]
        self.keys_dir = cfg["paths"]["keys_dir"]
        self.results_dir = cfg["paths"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

        # Repertoire public des clients (chapitre 9.3.1).
        with open(os.path.join(self.keys_dir, "public_directory.json")) as f:
            directory_b64 = json.load(f)
        self.public_keys = {
            int(cid): crypto.load_public_key(base64.b64decode(pub_b64))
            for cid, pub_b64 in directory_b64.items()
        }

        # Modele global (chapitre 2.2.4 / 6.1).
        mcfg = ModelConfig(
            d_in=cfg["dataset"]["n_features"],
            d_hidden=cfg["model"]["d_hidden"],
            rank=cfg["model"]["rank"],
            seed_base=cfg["model"]["seed_base"],
        )
        self.model_cfg = mcfg
        self.base_model = BaseModel(mcfg)
        rng0 = np.random.default_rng(mcfg.seed_base)
        self.global_adapter = LoRAAdapter(mcfg, rng0)

        # Jeu de test tenu par le serveur, pour EVALUATION SEULEMENT
        # (jamais utilise pendant l'entrainement -- voir docstring).
        test_path = os.path.join(cfg["paths"]["data_dir"], "test_set.npz")
        test_data = np.load(test_path)
        self.X_test, self.y_test = test_data["X"], test_data["y"]

        self.client_socks: Dict[int, ssl.SSLSocket] = {}
        self.client_n_k: Dict[int, int] = {}
        self.active_ids: List[int] = []
        self.dropped_ids: List[int] = []

        self.metrics_history: List[dict] = []

    # ------------------------------------------------------------------ #
    # Phase de configuration ("setup", chapitre 9.3.1)
    # ------------------------------------------------------------------ #
    def _accept_clients(self) -> None:
        ssl_ctx = net.make_server_ssl_context(
            os.path.join(self.keys_dir, "server_cert.pem"),
            os.path.join(self.keys_dir, "server_key.pem"),
        )
        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw_sock.bind((self.host, self.port))
        raw_sock.listen(self.n_clients)
        log(f"en ecoute sur {self.host}:{self.port} (TLS), attente de {self.n_clients} clients...")

        for _ in range(self.n_clients):
            conn, addr = raw_sock.accept()
            tls_conn = ssl_ctx.wrap_socket(conn, server_side=True)
            hello = net.recv_msg(tls_conn)
            assert hello["type"] == "HELLO"
            cid = int(hello["id"])
            self.client_socks[cid] = tls_conn
            self.client_n_k[cid] = int(hello["n_k"])
            log(f"client {cid} connecte (n_k={self.client_n_k[cid]} exemples locaux).")

        raw_sock.close()
        self.active_ids = sorted(self.client_socks.keys())

    def _broadcast_setup(self) -> None:
        n_total = sum(self.client_n_k.values())
        weights = {str(cid): self.client_n_k[cid] / n_total for cid in self.active_ids}
        for cid in self.active_ids:
            net.send_msg(self.client_socks[cid], {
                "type": "SETUP",
                "n_total": n_total,
                "weights": weights,
                "strategy": self.strategy,
                "privacy": self.cfg["privacy"],
                "security": self.cfg["security"],
                "federation": self.cfg["federation"],
                "model": {
                    "d_in": self.model_cfg.d_in, "d_hidden": self.model_cfg.d_hidden,
                    "rank": self.model_cfg.rank, "seed_base": self.model_cfg.seed_base,
                },
            })
        self.weights = weights
        log(f"phase de configuration terminee. n_total={n_total}, poids={weights}")

    # ------------------------------------------------------------------ #
    # Boucle de rounds (Algorithme 4.1)
    # ------------------------------------------------------------------ #
    def _broadcast_round_start(self, round_id: int) -> None:
        encoded_params = net.encode_param_dict(self.global_adapter.to_dict())
        for cid in self.active_ids:
            net.send_msg(self.client_socks[cid], {
                "type": "ROUND_START",
                "round_id": round_id,
                "active_ids": self.active_ids,
                "global_params": encoded_params,
            })

    def _collect_updates(self, round_id: int) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[int]]:
        updates: Dict[int, Dict[str, np.ndarray]] = {}
        newly_dropped: List[int] = []
        simulate = self.cfg["security"].get("simulate_dropout")

        for cid in list(self.active_ids):
            # Simulation deterministe d'un decrochage (pour DEMONTRER
            # l'algorithme 5.2 de facon reproductible, chapitre 8.4.3).
            if simulate and simulate["client_id"] == cid and simulate["round"] == round_id:
                log(f"(( simulation de decrochage du client {cid} au round {round_id} ))")
                try:
                    # On absorbe la mise a jour que le client envoie
                    # normalement (il ignore qu'il est "simule decroche"),
                    # afin de ne pas la laisser trainer dans le flux TCP.
                    self.client_socks[cid].settimeout(2.0)
                    _ = net.recv_msg(self.client_socks[cid])
                except Exception:
                    pass
                # On informe explicitement ce client qu'il doit s'arreter
                # proprement (sinon il resterait bloque en attente d'un
                # ROUND_START qui ne viendra plus, puisqu'il est retire de
                # `active_ids`).
                try:
                    net.send_msg(self.client_socks[cid], {"type": "DONE"})
                    self.client_socks[cid].close()
                except Exception:
                    pass
                newly_dropped.append(cid)
                continue

            try:
                self.client_socks[cid].settimeout(self.round_timeout)
                msg = net.recv_msg(self.client_socks[cid])
                assert msg["type"] == "MASKED_UPDATE" and msg["round_id"] == round_id
                updates[cid] = net.decode_param_dict(msg["delta"])
            except (socket.timeout, ConnectionError, OSError) as e:
                log(f"!! decrochage detecte pour le client {cid} au round {round_id} ({e})")
                newly_dropped.append(cid)

        return updates, newly_dropped

    def _request_shares_and_reconstruct(self, dropped_id: int, round_id: int):
        """Demande aux clients survivants leurs parts de Shamir de la cle
        privee du client `dropped_id` (Algorithme 5.2) et reconstruit sa
        cle privee des que le seuil est atteint."""
        threshold = self.cfg["security"]["shamir_threshold"]
        shares: List[Tuple[int, int]] = []

        for cid in self.active_ids:
            if cid == dropped_id or cid in self.dropped_ids:
                continue
            try:
                net.send_msg(self.client_socks[cid], {
                    "type": "REQUEST_SHARE", "owner": dropped_id, "round_id": round_id,
                })
                self.client_socks[cid].settimeout(self.round_timeout)
                resp = net.recv_msg(self.client_socks[cid])
                if resp["type"] == "SHARE_VALUE" and resp["owner"] == dropped_id:
                    shares.append((resp["share"][0], resp["share"][1]))
                    log(f"part de Shamir recue du client {cid} pour reconstruire le client {dropped_id}.")
            except Exception as e:
                log(f"impossible d'obtenir la part du client {cid} : {e}")

            if len(shares) >= threshold:
                break

        if len(shares) < threshold:
            log(f"!! ECHEC reconstruction client {dropped_id} : seulement {len(shares)}/{threshold} parts obtenues.")
            return None

        secret_int = crypto.shamir_reconstruct(shares)
        priv = crypto.int_to_private_key(secret_int)
        log(f"cle privee du client {dropped_id} reconstruite avec succes ({len(shares)} parts).")
        return priv

    def _apply_dropout_corrections(
        self, updates: Dict[int, Dict[str, np.ndarray]], newly_dropped: List[int], round_id: int,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Corrige les masques orphelins laisses par les clients decroches
        (Equation 5.4 + Algorithme 5.2) directement dans les mises a jour
        recues, en les redistribuant de facon additive sur les updates
        restantes (le resultat SOMME reste correct)."""
        if not newly_dropped or self.strategy != "secure_fedavg":
            return updates

        survivor_ids = [cid for cid in self.active_ids if cid not in newly_dropped]
        total_correction = {k: None for k in PARAM_KEYS}

        for dropped_id in newly_dropped:
            priv = self._request_shares_and_reconstruct(dropped_id, round_id)
            if priv is None:
                continue
            survivor_pubs = {cid: self.public_keys[cid] for cid in survivor_ids}
            mask_scale = self.cfg["security"]["mask_scale"]
            for k in PARAM_KEYS:
                shape = self.global_adapter.to_dict()[k].shape
                corr = crypto.compute_dropout_correction(
                    dropped_id, priv, survivor_ids, survivor_pubs, round_id, shape, mask_scale
                )
                total_correction[k] = corr if total_correction[k] is None else total_correction[k] + corr

        if all(v is None for v in total_correction.values()):
            return updates

        # On retranche la correction totale d'UNE seule mise a jour
        # survivante (peu importe laquelle : la somme finale est identique,
        # cf. linearite de la somme utilisee par fedavg_weighted_sum).
        first_survivor = survivor_ids[0]
        for k in PARAM_KEYS:
            if total_correction[k] is not None:
                updates[first_survivor][k] = updates[first_survivor][k] - total_correction[k]
        return updates

    def _aggregate(self, updates: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        ids = list(updates.keys())
        upd_list = [updates[i] for i in ids]
        weights_list = [self.client_n_k[i] for i in ids]

        if self.strategy == "secure_fedavg":
            # Les clients ont deja pondere puis masque leur update ; la
            # somme demasquee EST directement l'Eq. 4.2 (chapitre 9.3.2).
            return fedavg_weighted_sum(upd_list, weights_list)
        elif self.strategy == "plain_fedavg":
            return fedavg_plain(upd_list, weights_list)
        elif self.strategy == "plain_median":
            return robust_median(upd_list)
        elif self.strategy == "plain_trimmed":
            beta = self.cfg["federation"]["trimmed_beta"]
            return robust_trimmed_mean(upd_list, beta)
        else:
            raise ValueError(f"Strategie d'agregation inconnue : {self.strategy}")

    def _evaluate(self, round_id: int, t_round_start: float) -> dict:
        p_pred = self.global_adapter.predict_proba(self.base_model, self.X_test)
        m = compute_metrics(self.y_test, p_pred)

        # --- Calcul du budget epsilon cumule a ce round (DP-SGD, Eq. 5.6) ---
        pcfg = PrivacyConfig(
            enabled=self.cfg["privacy"]["enabled"],
            clip_norm=self.cfg["privacy"]["clip_norm"],
            noise_multiplier=self.cfg["privacy"]["noise_multiplier"],
            delta=self.cfg["privacy"]["delta"],
            sampling_rate=self.cfg["privacy"]["sampling_rate"],
        )
        # steps = round_id * (local_epochs effectifs) ; ici 1 pas DP par round suffit
        # (on peut raffiner en multipliant par local_epochs si le client fait plusieurs pas)
        n_steps = round_id
        epsilon = estimate_epsilon(pcfg, steps=n_steps) if pcfg.enabled else 0.0

        record = {
            "round": round_id,
            "n_active_clients": len(self.active_ids),
            "duration_s": round(time.time() - t_round_start, 3),
            **m.as_dict(),
            "epsilon": round(epsilon, 4),   # <-- NOUVELLE COLONNE
        }
        self.metrics_history.append(record)

        log(f"round {round_id:3d}/{self.n_rounds} | actifs={len(self.active_ids):2d} | "
            f"{m} | eps={epsilon:.3f} | duree={record['duration_s']}s")
        return record

    def _save_results(self) -> None:
        # Modele final "pret pour la production" (chapitre 9.3.3).
        model_path = os.path.join(self.results_dir, "global_model_final.npz")
        gd = self.global_adapter.to_dict()
        np.savez(model_path, **gd, d_in=self.model_cfg.d_in, d_hidden=self.model_cfg.d_hidden,
                 rank=self.model_cfg.rank, seed_base=self.model_cfg.seed_base)
        log(f"modele global final sauvegarde dans {model_path}")

        # Historique des metriques (pour le chapitre 11 du memoire).
        csv_path = os.path.join(self.results_dir, "metrics_history.csv")
        if self.metrics_history:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.metrics_history[0].keys()))
                writer.writeheader()
                writer.writerows(self.metrics_history)
            log(f"historique des metriques sauvegarde dans {csv_path}")

    def run(self) -> None:
        self._accept_clients()
        self._broadcast_setup()

        for round_id in range(1, self.n_rounds + 1):
            t0 = time.time()
            self._broadcast_round_start(round_id)
            updates, newly_dropped = self._collect_updates(round_id)

            if newly_dropped:
                updates = self._apply_dropout_corrections(updates, newly_dropped, round_id)
                self.dropped_ids.extend(newly_dropped)
                self.active_ids = [c for c in self.active_ids if c not in newly_dropped]

            if not updates:
                log("!! aucune mise a jour recue ce round, arret anticipe.")
                break

            aggregated_delta = self._aggregate(updates)
            self.global_adapter.A = self.global_adapter.A + aggregated_delta["A"]
            self.global_adapter.B = self.global_adapter.B + aggregated_delta["B"]
            self.global_adapter.w_out = self.global_adapter.w_out + aggregated_delta["w_out"]
            self.global_adapter.b_out = self.global_adapter.b_out + float(
                np.asarray(aggregated_delta["b_out"]).reshape(-1)[0]
            )

            self._evaluate(round_id, t0)

        for cid in self.active_ids:
            try:
                net.send_msg(self.client_socks[cid], {"type": "DONE"})
                self.client_socks[cid].close()
            except Exception:
                pass

        self._save_results()
        log("entrainement federe termine.")


def main():
    parser = argparse.ArgumentParser(description="Serveur federateur (chapitre 9.3 du memoire).")
    parser.add_argument("--config", default="config/base_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    server = FederatedServer(cfg)
    server.run()


if __name__ == "__main__":
    main()