"""
net_protocol.py
================
Couche de transport du prototype (chapitre 9, "flots de donnees et de
mises a jour" ; chapitre 10.3.4, "securite et authentification").

- Chiffrement en TRANSIT (chapitre 5.3.1) : toutes les connexions passent
  par TLS (module `ssl`), a l'aide d'un certificat auto-signe genere par
  `key_generate.py`.
- Format de message : chaque message est un objet JSON precede de sa
  longueur sur 8 octets (big-endian), ce qui permet de savoir exactement
  combien d'octets lire cote reception (evite les messages tronques ou
  concatenes sur un flux TCP).
- Les tableaux NumPy (parametres du modele, mises a jour masquees) sont
  encodes en base64 (bytes bruts float64 + forme + dtype) plutot qu'en
  listes JSON, ce qui est nettement plus compact et plus rapide.
"""

from __future__ import annotations
import json
import socket
import ssl
import struct
import base64
from typing import Any, Dict

import numpy as np


# -------------------------------------------------------------------- #
# (De)serialisation des tableaux NumPy en JSON
# -------------------------------------------------------------------- #
def _ndarray_to_jsonable(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "__ndarray__": True,
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _jsonable_to_ndarray(d: Dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(d["data"])
    arr = np.frombuffer(raw, dtype=np.float64).reshape(d["shape"])
    return arr.copy()  # copy : frombuffer retourne un tableau en lecture seule


def encode_param_dict(params: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {k: _ndarray_to_jsonable(v) for k, v in params.items()}


def decode_param_dict(d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: _jsonable_to_ndarray(v) for k, v in d.items()}


# -------------------------------------------------------------------- #
# Envoi / reception de messages prefixes par leur longueur
# -------------------------------------------------------------------- #
def send_msg(sock: socket.socket, obj: Dict[str, Any]) -> None:
    payload = json.dumps(obj).encode("utf-8")
    header = struct.pack(">Q", len(payload))
    sock.sendall(header + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connexion fermee avant reception complete du message.")
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> Dict[str, Any]:
    header = _recv_exact(sock, 8)
    (length,) = struct.unpack(">Q", header)
    payload = _recv_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


# -------------------------------------------------------------------- #
# Contextes TLS (chapitre 5.3.1 : chiffrement en transit)
# -------------------------------------------------------------------- #
def make_server_ssl_context(certfile: str, keyfile: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return ctx


def make_client_ssl_context(ca_certfile: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=ca_certfile)
    ctx.check_hostname = False  # certificat auto-signe local (pas de nom DNS reel)
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


if __name__ == "__main__":
    # Auto-test : round-trip d'un dict de parametres a travers le codec JSON.
    original = {
        "A": np.random.default_rng(0).normal(size=(3, 4)),
        "B": np.random.default_rng(1).normal(size=(5, 3)),
    }
    encoded = encode_param_dict(original)
    decoded = decode_param_dict(encoded)
    for k in original:
        assert np.allclose(original[k], decoded[k]), f"Round-trip incorrect pour {k}"
    print("[net_protocol.py] round-trip encode/decode des tableaux NumPy : OK")

    # Auto-test : envoi/reception via une paire de sockets locaux (sans TLS).
    import threading

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)

    received = {}

    def _server_thread():
        conn, _ = srv.accept()
        received["msg"] = recv_msg(conn)
        conn.close()

    t = threading.Thread(target=_server_thread)
    t.start()

    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    send_msg(cli, {"type": "TEST", "params": encoded})
    cli.close()
    t.join()
    srv.close()

    assert received["msg"]["type"] == "TEST"
    print("[net_protocol.py] envoi/reception de message via socket local : OK")
    print("[net_protocol.py] auto-test OK.")
