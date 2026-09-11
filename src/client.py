"""
client.py
=========
Client federe (chapitre 9.2 du memoire). Execute, a chaque round :

  1) Reception du modele global courant (adaptateurs LoRA A, B, w_out, b_out).
  2) Entrainement LOCAL avec confidentialite differentielle (Algorithme 5.1,
     DP-SGD) : plusieurs epoques de descente de gradient, avec ecretage
     PAR EXEMPLE et ajout de bruit gaussien -- c'est ICI que le bruit est
     ajoute, cote client, AVANT tout envoi (jamais cote serveur).
  3) Calcul de la mise a jour ("pseudo-gradient") delta = params_locaux -
     params_globaux_recus, ponderee par le poids FedAvg n_k/n (Eq. 4.2).
  4) MASQUAGE additif de cette mise a jour deja bruitee (Equation 5.4,
     agregation securisee) : c'est l'equivalent, dans ce prototype, du
     "chiffrement du gradient" avant envoi -- le serveur ne voit jamais
     la mise a jour en clair, seule la SOMME totale (une fois demasquee)
     est recuperable, jamais une contribution individuelle.
  5) Envoi de la mise a jour bruitee ET masquee au serveur.
  6) Reponse, si sollicite, aux demandes de revelation de part de Shamir
     (Algorithme 5.2), pour permettre au serveur de corriger l'agregat en
     cas de decrochage d'un AUTRE client.

Usage :
    python client.py --id 1 --config config/base_config.yaml
"""

from __future__ import annotations
import argparse
import base64
import json
import os
import socket
import ssl
import time
from typing import Dict, Optional, Tuple, List

import numpy as np
import yaml

import crypto
import net_protocol as net
from model_build import ModelConfig, BaseModel, LoRAAdapter, PARAM_KEYS
from privacy import PrivacyConfig, dp_sgd_step


def log(cid: int, msg: str) -> None:
    print(f"[client {cid}] {msg}", flush=True)


class FederatedClient:
    def __init__(self, client_id: int, cfg: dict):
        self.id = client_id
        self.cfg = cfg
        self.keys_dir = cfg["paths"]["keys_dir"]
        self.data_dir = cfg["paths"]["data_dir"]
        self._pending_msg: Optional[dict] = None

        # --- Cles cryptographiques propres a ce client ---------------------
        with open(os.path.join(self.keys_dir, f"client_{client_id}_priv.bin"), "rb") as f:
            self.priv_key = crypto.load_private_key(f.read())
        with open(os.path.join(self.keys_dir, "public_directory.json")) as f:
            directory_b64 = json.load(f)
        self.public_keys = {
            int(cid): crypto.load_public_key(base64.b64decode(pub_b64))
            for cid, pub_b64 in directory_b64.items()
        }

        # --- Parts de Shamir DONT ce client est porteur (Algorithme 5.2) --
        # Chaque fichier shares/for_<self>_of_<owner>.enc est dechiffre ICI,
        # une seule fois, avec le secret ECDH partage entre `owner` et soi.
        self.held_shares: Dict[int, Tuple[int, int]] = {}
        shares_dir = os.path.join(self.keys_dir, "shares")
        prefix = f"for_{client_id}_of_"
        if os.path.isdir(shares_dir):
            for fname in os.listdir(shares_dir):
                if fname.startswith(prefix) and fname.endswith(".enc"):
                    owner_id = int(fname[len(prefix): -len(".enc")])
                    with open(os.path.join(shares_dir, fname)) as f:
                        token = f.read()
                    shared_secret = crypto.ecdh_shared_secret(self.priv_key, self.public_keys[owner_id])
                    context = f"shamir-share|owner={owner_id}|holder={client_id}"
                    self.held_shares[owner_id] = crypto.decrypt_share(shared_secret, context, token)
        log(self.id, f"parts de Shamir detenues pour les clients : {list(self.held_shares.keys())}")

        # --- Donnees locales (chapitre 9.2.1) ------------------------------
        shard_path = os.path.join(self.data_dir, f"client_{client_id}.npz")
        shard = np.load(shard_path)
        self.X, self.y = shard["X"], shard["y"]
        log(self.id, f"jeu de donnees local charge : {len(self.X)} exemples.")

        self.sock: Optional[ssl.SSLSocket] = None

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        net_cfg = self.cfg["network"]
        ssl_ctx = net.make_client_ssl_context(os.path.join(self.keys_dir, "server_cert.pem"))
        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_sock.settimeout(net_cfg["connect_timeout_s"])
        raw_sock.connect((net_cfg["host"], net_cfg["port"]))
        self.sock = ssl_ctx.wrap_socket(raw_sock, server_hostname="localhost")
        self.sock.settimeout(None)
        net.send_msg(self.sock, {"type": "HELLO", "id": self.id, "n_k": len(self.X)})
        log(self.id, "connecte au serveur (TLS), HELLO envoye.")

    def wait_setup(self) -> dict:
        msg = net.recv_msg(self.sock)
        assert msg["type"] == "SETUP"
        self.n_total = msg["n_total"]
        self.weight = msg["weights"][str(self.id)]
        self.strategy = msg["strategy"]
        self.privacy_cfg = PrivacyConfig(**msg["privacy"])
        self.security_cfg = msg["security"]
        self.fed_cfg = msg["federation"]
        mcfg = ModelConfig(**msg["model"])
        self.model_cfg = mcfg
        self.base_model = BaseModel(mcfg)
        log(self.id, f"configuration recue. poids FedAvg = {self.weight:.4f}")
        return msg

    # ------------------------------------------------------------------ #
    # Entrainement local avec DP-SGD (Algorithme 5.1)
    # ------------------------------------------------------------------ #
    def local_train(self, global_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        adapter = LoRAAdapter(self.model_cfg)
        adapter.load_dict(global_params)

        rng = np.random.default_rng(self.id * 10_000 + int(time.time() * 1000) % 100_000)
        epochs = self.fed_cfg["local_epochs"]
        batch_size = self.fed_cfg["local_batch_size"]
        lr = self.fed_cfg["learning_rate"]
        n = len(self.X)

        for epoch in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                Xb, yb = self.X[idx], self.y[idx]
                if len(Xb) == 0:
                    continue
                grads = adapter.per_example_gradients(self.base_model, Xb, yb)
                if self.privacy_cfg.enabled:
                    step = dp_sgd_step(grads, self.privacy_cfg, batch_size=len(Xb), rng=rng)
                else:
                    step = {
                        "A": grads["grad_A"].mean(axis=0), "B": grads["grad_B"].mean(axis=0),
                        "w_out": grads["grad_w_out"].mean(axis=0), "b_out": np.array([grads["grad_b_out"].mean()]),
                    }
                adapter.apply_update(
                    {"grad_A": step["A"], "grad_B": step["B"],
                     "grad_w_out": step["w_out"], "grad_b_out": float(step["b_out"][0])},
                    lr=lr,
                )

        local_params = adapter.to_dict()
        delta = {k: local_params[k] - global_params[k] for k in PARAM_KEYS}
        return delta

    def _weight_and_mask(self, delta: Dict[str, np.ndarray], peer_ids: List[int], round_id: int) -> Dict[str, np.ndarray]:
        weighted = {k: delta[k] * self.weight for k in PARAM_KEYS}
        if self.strategy != "secure_fedavg":
            return weighted  # pas de masquage pour les strategies "plain_*"

        mask_scale = self.security_cfg["mask_scale"]
        masked = {}
        for k in PARAM_KEYS:
            masked[k] = crypto.mask_update(
                self.id, peer_ids, self.priv_key,
                {j: self.public_keys[j] for j in peer_ids if j != self.id},
                round_id, weighted[k], mask_scale,
            )
        return masked

    def _next_message(self) -> dict:
        """Retourne le prochain message logique : soit un message deja mis
        en attente par `_answer_pending_share_requests`, soit le prochain
        message lu sur le socket."""
        if self._pending_msg is not None:
            msg = self._pending_msg
            self._pending_msg = None
            return msg
        return net.recv_msg(self.sock)

    def _answer_pending_share_requests(self) -> None:
        """Repond a toute rafale de REQUEST_SHARE envoyee par le serveur
        avant le round suivant (Algorithme 5.2). Des qu'un message d'un
        AUTRE type arrive (ROUND_START ou DONE), il est memorise dans
        `self._pending_msg` pour etre consomme par `_next_message`."""
        self.sock.settimeout(2.0)
        try:
            while True:
                try:
                    msg = net.recv_msg(self.sock)
                except socket.timeout:
                    return
                if msg["type"] == "REQUEST_SHARE":
                    owner = msg["owner"]
                    if owner in self.held_shares:
                        x, y = self.held_shares[owner]
                        net.send_msg(self.sock, {"type": "SHARE_VALUE", "owner": owner, "share": [x, y]})
                        log(self.id, f"part de Shamir revelee pour le client decroche {owner}.")
                    else:
                        net.send_msg(self.sock, {"type": "SHARE_VALUE", "owner": owner, "share": None})
                else:
                    self._pending_msg = msg
                    return
        finally:
            self.sock.settimeout(None)

    # ------------------------------------------------------------------ #
    # AJOUT : log des metriques locales par round (pour le dashboard)
    # ------------------------------------------------------------------ #
    def _log_round_metrics(self, round_id: int, grad_norm: float, duration: float) -> None:
        """Ecrit les metriques locales de ce round dans logs/client_<id>_metrics.csv
        (utilise par le dashboard pour le suivi par client)."""
        import csv as _csv
        logs_dir = self.cfg["paths"]["logs_dir"]
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, f"client_{self.id}_metrics.csv")
        file_exists = os.path.isfile(path)
        try:
            with open(path, "a", newline="") as f:
                w = _csv.writer(f)
                if not file_exists:
                    w.writerow(["round", "grad_norm", "duration_s", "status"])
                w.writerow([round_id, round(grad_norm, 4), round(duration, 4), "active"])
        except Exception as e:
            log(self.id, f"!! echec log metriques : {e}")

    # ------------------------------------------------------------------ #
    # Boucle principale de rounds
    # ------------------------------------------------------------------ #
    def run_rounds(self) -> None:
        while True:
            msg = self._next_message()

            if msg["type"] == "DONE":
                log(self.id, "entrainement federe termine (message DONE recu).")
                break

            assert msg["type"] == "ROUND_START", f"Message inattendu : {msg['type']}"
            round_id = msg["round_id"]
            active_ids = msg["active_ids"]
            global_params = net.decode_param_dict(msg["global_params"])

            t0 = time.time()
            delta = self.local_train(global_params)
            masked_delta = self._weight_and_mask(delta, active_ids, round_id)

            # --- AJOUT : norme L2 globale du delta (pseudo-gradient) avant envoi ---
            flat_delta = np.concatenate([delta[k].reshape(-1) for k in PARAM_KEYS])
            grad_norm = float(np.linalg.norm(flat_delta))

            net.send_msg(self.sock, {
                "type": "MASKED_UPDATE", "round_id": round_id, "id": self.id,
                "delta": net.encode_param_dict(masked_delta),
            })
            duration = time.time() - t0
            log(self.id, f"round {round_id} : mise a jour locale envoyee "
                         f"(entrainement+masquage en {duration:.2f}s).")

            # --- AJOUT : ecriture CSV des metriques locales ---
            self._log_round_metrics(round_id, grad_norm, duration)

            self._answer_pending_share_requests()

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Client federe (chapitre 9.2 du memoire).")
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--config", default="config/base_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    client = FederatedClient(args.id, cfg)
    client.connect()
    client.wait_setup()
    client.run_rounds()
    client.close()


if __name__ == "__main__":
    main()