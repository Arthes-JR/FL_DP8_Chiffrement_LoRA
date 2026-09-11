"""
key_generate.py
================
Script de generation des cles et certificats, execute UNE FOIS avant le
lancement de l'experience (phase de configuration / "setup", chapitre
9.3.1 du memoire).

Produit, dans le dossier `keys/` :
  - server_cert.pem, server_key.pem
        Certificat TLS auto-signe du serveur (chiffrement en transit,
        chapitre 5.3.1). A utiliser uniquement pour la demonstration :
        un deploiement reel utiliserait une autorite de certification
        interne a l'organisation ou reconnue publiquement.

  - client_<id>_priv.bin, client_<id>_pub.bin
        Paire de cles X25519 de chaque client (Diffie-Hellman, chapitre
        5.2.1), utilisees pour deriver les secrets partagees s_ij du
        masquage additif (Equation 5.4).

  - public_directory.json
        Repertoire public {id -> cle publique en base64}. Dans un
        deploiement reel, ce repertoire serait construit dynamiquement par
        le serveur agissant comme relais pendant la phase de setup
        (chapitre 9.3.1) ; il est ici precalcule hors ligne pour simplifier
        le lancement de la simulation locale (limite assumee, voir
        README.md).

  - shares/for_<holder>_of_<owner>.enc
        Part de Shamir (chapitre 5.2.2, Algorithme 5.2) de la cle privee
        du client `owner`, destinee au client `holder`, CHIFFREE avec la
        cle symetrique derivee du secret ECDH partage entre `owner` et
        `holder` -- de sorte que le serveur, qui relaie/stocke ces fichiers,
        ne puisse pas les lire : seul `holder` peut les dechiffrer.

Usage :
    python key_generate.py --config config/base_config.yaml
"""

from __future__ import annotations
import argparse
import base64
import datetime
import json
import os
import shutil

import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import crypto


def generate_server_tls_cert(cert_path: str, key_path: str, common_name: str = "fedserver") -> None:
    """Genere un certificat TLS auto-signe (RSA 2048) pour le serveur federe."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


def generate_all_keys(n_clients: int, threshold: int, out_dir: str) -> None:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    shares_dir = os.path.join(out_dir, "shares")
    os.makedirs(shares_dir, exist_ok=True)

    # --- 1) Certificat TLS du serveur -----------------------------------
    generate_server_tls_cert(
        os.path.join(out_dir, "server_cert.pem"), os.path.join(out_dir, "server_key.pem")
    )
    print(f"[key_generate] certificat TLS serveur genere dans {out_dir}/server_cert.pem")

    # --- 2) Paires de cles X25519 par client -----------------------------
    client_ids = list(range(1, n_clients + 1))
    privs, pubs = {}, {}
    for cid in client_ids:
        priv, pub = crypto.generate_keypair()
        privs[cid] = priv
        pubs[cid] = pub
        with open(os.path.join(out_dir, f"client_{cid}_priv.bin"), "wb") as f:
            f.write(crypto.serialize_private_key(priv))
        with open(os.path.join(out_dir, f"client_{cid}_pub.bin"), "wb") as f:
            f.write(crypto.serialize_public_key(pub))
    print(f"[key_generate] {n_clients} paires de cles X25519 generees.")

    # --- 3) Repertoire public ---------------------------------------------
    directory = {
        str(cid): base64.b64encode(crypto.serialize_public_key(pubs[cid])).decode("ascii")
        for cid in client_ids
    }
    with open(os.path.join(out_dir, "public_directory.json"), "w") as f:
        json.dump(directory, f, indent=2)
    print(f"[key_generate] repertoire public ecrit dans {out_dir}/public_directory.json")

    # --- 4) Partage de Shamir de chaque cle privee, chiffre par paire -----
    n_holders = n_clients - 1
    if n_holders < threshold:
        raise ValueError(
            f"Seuil de reconstruction ({threshold}) > nombre de porteurs de parts possibles "
            f"({n_holders}). Reduisez le seuil ou augmentez le nombre de clients."
        )

    for owner in client_ids:
        holders = [c for c in client_ids if c != owner]
        secret_int = crypto.private_key_to_int(privs[owner])
        shares = crypto.shamir_split(secret_int, n=len(holders), t=threshold)
        for holder, share in zip(holders, shares):
            shared_secret = crypto.ecdh_shared_secret(privs[owner], pubs[holder])
            context = f"shamir-share|owner={owner}|holder={holder}"
            token = crypto.encrypt_share(shared_secret, context, share)
            fname = os.path.join(shares_dir, f"for_{holder}_of_{owner}.enc")
            with open(fname, "w") as f:
                f.write(token)
    print(f"[key_generate] parts de Shamir (seuil={threshold}) generees et chiffrees dans {shares_dir}/")

    # --- 5) Manifeste ------------------------------------------------------
    manifest = {
        "n_clients": n_clients,
        "threshold": threshold,
        "client_ids": client_ids,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[key_generate] manifest.json ecrit. Generation des cles terminee.")


def main():
    parser = argparse.ArgumentParser(description="Genere les cles et certificats du prototype federe.")
    parser.add_argument("--config", default="config/base_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generate_all_keys(
        n_clients=cfg["federation"]["n_clients"],
        threshold=cfg["security"]["shamir_threshold"],
        out_dir=cfg["paths"]["keys_dir"],
    )


if __name__ == "__main__":
    main()
