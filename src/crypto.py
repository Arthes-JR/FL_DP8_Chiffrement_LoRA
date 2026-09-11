"""
crypto.py
=========
Primitives cryptographiques du prototype, implementant :

 1) L'echange de cles Diffie-Hellman sur courbe X25519 entre chaque paire
    de clients (chapitre 5.2.1 : "cle secrete partagee s_ij").
 2) La derivation d'une graine de generateur pseudo-aleatoire (PRG) a partir
    du secret partage, via HKDF (norme RFC 5869).
 3) Le masquage additif des mises a jour (Equation 5.4 du memoire) :

        y_i = x_i + sum_{j>i} s_ij - sum_{j<i} s_ji

    implemente ici directement en arithmetique flottante (les masques
    s'annulent exactement des lors que chaque paire (i,j) utilise LA MEME
    graine des deux cotes, ce qui est garanti par la propriete de symetrie
    de Diffie-Hellman : ECDH(priv_i, pub_j) == ECDH(priv_j, pub_i)).

 4) Le partage de secret de Shamir (t-parmi-n), utilise pour permettre la
    reconstruction de la cle privee d'un client qui decroche en cours de
    round (Algorithme 5.2 : "les clients survivants revelent les parts de
    secret permettant de retirer la contribution du client absent").

 5) Un chiffrement symetrique (Fernet/AES) pour transporter chaque part de
    secret de maniere opaque au serveur (le serveur relaie les parts sans
    pouvoir les lire tant que le client correspondant n'a pas decroche).

REMARQUE DE RIGUEUR (cf. chapitre 5.2.3 du memoire, "cout computationnel") :
Ce module est une IMPLEMENTATION PEDAGOGIQUE fidele aux idees de Bonawitz
et al. (2017), mais simplifiee : pas de gestion de plusieurs decrochages
simultanes avec preuve de securite formelle, pas d'arithmetique modulaire
sur un corps fini pour le masquage (utilisation directe de flottants IEEE
754, suffisante pour une demonstration mais pas pour une preuve
d'indistinguabilite rigoureuse). Le partage de Shamir, lui, EST implemente
sur un corps premier fini et est cryptographiquement correct.
"""

from __future__ import annotations
import os
import secrets
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.fernet import Fernet
import base64


# ========================================================================
# 1) Paires de cles X25519 (Diffie-Hellman sur courbe elliptique)
# ========================================================================

def generate_keypair() -> Tuple[x25519.X25519PrivateKey, x25519.X25519PublicKey]:
    priv = x25519.X25519PrivateKey.generate()
    return priv, priv.public_key()


def serialize_private_key(priv: x25519.X25519PrivateKey) -> bytes:
    return priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def serialize_public_key(pub: x25519.X25519PublicKey) -> bytes:
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )


def load_private_key(raw: bytes) -> x25519.X25519PrivateKey:
    return x25519.X25519PrivateKey.from_private_bytes(raw)


def load_public_key(raw: bytes) -> x25519.X25519PublicKey:
    return x25519.X25519PublicKey.from_public_bytes(raw)


# ========================================================================
# 2) Derivation de graine PRG a partir d'un secret partage (HKDF)
# ========================================================================

def ecdh_shared_secret(my_priv: x25519.X25519PrivateKey, their_pub: x25519.X25519PublicKey) -> bytes:
    """
    Calcule le secret partage ECDH. Par symetrie de Diffie-Hellman :
        ecdh_shared_secret(priv_i, pub_j) == ecdh_shared_secret(priv_j, pub_i)
    C'est cette propriete qui garantit que les DEUX clients d'une paire
    (i, j) derivent EXACTEMENT le meme masque, condition necessaire a
    l'annulation de l'Equation 5.4.
    """
    return my_priv.exchange(their_pub)


def derive_seed(shared_secret: bytes, context: str) -> int:
    """
    Derive une graine entiere 64 bits a partir d'un secret partage, via
    HKDF-SHA256 (RFC 5869). `context` doit encoder de maniere unique la
    paire de clients ET le round courant, afin que chaque round utilise un
    masque INDEPENDANT (un masque reutilise entre rounds fuiterait de
    l'information sur la difference des mises a jour successives).
    """
    hkdf = HKDF(algorithm=hashes.SHA256(), length=8, salt=None, info=context.encode("utf-8"))
    derived = hkdf.derive(shared_secret)
    return int.from_bytes(derived, "big")


def generate_mask(seed: int, shape: Tuple[int, ...], scale: float = 1.0) -> np.ndarray:
    """Genere un masque pseudo-aleatoire DETERMINISTE a partir d'une graine."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=shape)


# ========================================================================
# 3) Masquage / demasquage additif d'une mise a jour (Equation 5.4)
# ========================================================================

def compute_pairwise_mask(
    my_id: int, peer_id: int, my_priv: x25519.X25519PrivateKey,
    peer_pub: x25519.X25519PublicKey, round_id: int, shape: Tuple[int, ...],
    mask_scale: float = 1.0,
) -> np.ndarray:
    """
    Calcule s_ij pour la paire (my_id, peer_id) au round `round_id`,
    avec une forme `shape` identique a celle du vecteur a masquer.
    """
    secret = ecdh_shared_secret(my_priv, peer_pub)
    lo, hi = min(my_id, peer_id), max(my_id, peer_id)
    context = f"secagg|pair={lo}-{hi}|round={round_id}"
    seed = derive_seed(secret, context)
    return generate_mask(seed, shape, scale=mask_scale)


def mask_update(
    my_id: int, peer_ids: List[int], my_priv: x25519.X25519PrivateKey,
    peer_pubs: Dict[int, x25519.X25519PublicKey], round_id: int,
    update: np.ndarray, mask_scale: float = 1.0,
) -> np.ndarray:
    """
    Applique l'Equation 5.4 :  y_i = x_i + sum_{j>i} s_ij - sum_{j<i} s_ji
    """
    y = update.copy()
    for j in peer_ids:
        if j == my_id:
            continue
        s_ij = compute_pairwise_mask(my_id, j, my_priv, peer_pubs[j], round_id, update.shape, mask_scale)
        if j > my_id:
            y = y + s_ij
        else:
            y = y - s_ij
    return y


def compute_dropout_correction(
    dropped_id: int, dropped_priv: x25519.X25519PrivateKey, survivor_ids: List[int],
    survivor_pubs: Dict[int, x25519.X25519PublicKey], round_id: int, shape: Tuple[int, ...],
    mask_scale: float = 1.0,
) -> np.ndarray:
    """
    Calcule la CORRECTION a appliquer a la somme des mises a jour recues
    lorsque le client `dropped_id` a decroche AVANT d'envoyer sa propre
    mise a jour masquee.

    Pour chaque survivant j : le terme s_{dropped,j} est resté "orphelin"
    dans y_j (car le cote du client `dropped_id` n'a jamais ete envoye).
    On reconstruit s_{dropped,j} = ECDH(priv_dropped, pub_j) et on l'annule
    avec le signe oppose a celui utilise par j :
      - si j > dropped_id : j avait AJOUTE +s_ij a son y_j si dropped<j... 
        (voir mask_update : le cote "j" ajoute quand j>my_id, ici my_id=j,
        peer=dropped ; donc j ajoute +s si dropped>j, soustrait si dropped<j)
    Cette fonction recalcule directement le signe correct pour chaque
    survivant et retourne la somme des corrections a RETRANCHER de la somme
    recue (voir server.py::_apply_dropout_correction pour l'usage exact).
    """
    correction = np.zeros(shape)
    for j in survivor_ids:
        s = compute_pairwise_mask(dropped_id, j, dropped_priv, survivor_pubs[j], round_id, shape, mask_scale)
        # Dans mask_update, le client j (peer=dropped_id) applique :
        #   +s si dropped_id > j   (car alors "peer > my_id" du point de vue de j)
        #   -s si dropped_id < j
        # C'est exactement ce terme, encore present dans la somme recue,
        # qu'il faut retrancher.
        if dropped_id > j:
            correction = correction + s
        else:
            correction = correction - s
    return correction


# ========================================================================
# 4) Partage de secret de Shamir (t-parmi-n) sur un corps premier fini
# ========================================================================

# Nombre premier de Mersenne (2^521 - 1), largement plus grand que l'espace
# des cles privees X25519 (32 octets = 2^256), pour un partage sans perte.
_PRIME = (1 << 521) - 1


def _eval_poly(coeffs: List[int], x: int, prime: int = _PRIME) -> int:
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % prime
    return result


def shamir_split(secret_int: int, n: int, t: int, prime: int = _PRIME) -> List[Tuple[int, int]]:
    """
    Partage `secret_int` en n parts, dont t suffisent a le reconstruire
    (polynome aleatoire de degre t-1 avec terme constant = secret).
    """
    if secret_int >= prime:
        raise ValueError("Le secret doit etre strictement inferieur au nombre premier utilise.")
    coeffs = [secret_int] + [secrets.randbelow(prime) for _ in range(t - 1)]
    shares = [(x, _eval_poly(coeffs, x, prime)) for x in range(1, n + 1)]
    return shares


def _mod_inverse(a: int, prime: int) -> int:
    return pow(a, prime - 2, prime)  # petit theoreme de Fermat (prime est premier)


def shamir_reconstruct(shares: List[Tuple[int, int]], prime: int = _PRIME) -> int:
    """Reconstruit le secret via interpolation de Lagrange en x=0."""
    secret = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i == j:
                continue
            num = (num * (-xj)) % prime
            den = (den * (xi - xj)) % prime
        lagrange_coeff = (num * _mod_inverse(den % prime, prime)) % prime
        secret = (secret + yi * lagrange_coeff) % prime
    return secret % prime


def private_key_to_int(priv: x25519.X25519PrivateKey) -> int:
    return int.from_bytes(serialize_private_key(priv), "big")


def int_to_private_key(value: int) -> x25519.X25519PrivateKey:
    raw = value.to_bytes(32, "big")
    return load_private_key(raw)


# ========================================================================
# 5) Chiffrement symetrique des parts (opaque pour le serveur-relais)
# ========================================================================

def derive_fernet_key(shared_secret: bytes, context: str) -> bytes:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=context.encode("utf-8"))
    return base64.urlsafe_b64encode(hkdf.derive(shared_secret))


def encrypt_share(shared_secret: bytes, context: str, share: Tuple[int, int]) -> str:
    key = derive_fernet_key(shared_secret, context)
    f = Fernet(key)
    payload = f"{share[0]}:{share[1]}".encode("utf-8")
    return f.encrypt(payload).decode("utf-8")


def decrypt_share(shared_secret: bytes, context: str, token: str) -> Tuple[int, int]:
    key = derive_fernet_key(shared_secret, context)
    f = Fernet(key)
    payload = f.decrypt(token.encode("utf-8")).decode("utf-8")
    x_str, y_str = payload.split(":")
    return int(x_str), int(y_str)


if __name__ == "__main__":
    # ---- Auto-test 1 : les masques s'annulent (Equation 5.4) ----
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()

    shape = (10,)
    x_a = np.random.default_rng(1).normal(size=shape)
    x_b = np.random.default_rng(2).normal(size=shape)

    y_a = mask_update(my_id=1, peer_ids=[1, 2], my_priv=priv_a, peer_pubs={2: pub_b}, round_id=0, update=x_a)
    y_b = mask_update(my_id=2, peer_ids=[1, 2], my_priv=priv_b, peer_pubs={1: pub_a}, round_id=0, update=x_b)

    recovered_sum = y_a + y_b
    true_sum = x_a + x_b
    err = np.max(np.abs(recovered_sum - true_sum))
    print(f"[crypto.py] erreur max apres demasquage (2 clients) = {err:.2e}")
    assert err < 1e-9, "Les masques doivent s'annuler exactement."

    # ---- Auto-test 2 : partage de Shamir (5 parts, seuil 3) ----
    secret_val = private_key_to_int(priv_a)
    shares = shamir_split(secret_val, n=5, t=3)
    subset = shares[:3]  # exactement le seuil
    reconstructed = shamir_reconstruct(subset)
    print(f"[crypto.py] secret original == secret reconstruit (Shamir) : {reconstructed == secret_val}")
    assert reconstructed == secret_val, "Reconstruction de Shamir incorrecte."

    # ---- Auto-test 3 : chiffrement opaque d'une part ----
    shared = ecdh_shared_secret(priv_a, pub_b)
    token = encrypt_share(shared, "share|client=1|for=2", shares[0])
    dec = decrypt_share(shared, "share|client=1|for=2", token)
    assert dec == shares[0], "Chiffrement/dechiffrement de part incorrect."
    print("[crypto.py] chiffrement/dechiffrement de part OK.")

    # ---- Auto-test 4 : correction de decrochage (3 clients, 1 decroche) ----
    priv_c, pub_c = generate_keypair()
    ids = [1, 2, 3]
    pubs = {1: pub_a, 2: pub_b, 3: pub_c}
    privs = {1: priv_a, 2: priv_b, 3: priv_c}
    xs = {i: np.random.default_rng(10 + i).normal(size=shape) for i in ids}

    ys = {
        i: mask_update(i, ids, privs[i], {j: pubs[j] for j in ids if j != i}, round_id=7, update=xs[i])
        for i in ids
    }
    # Le client 2 decroche : le serveur ne recoit que y_1 et y_3.
    received_sum = ys[1] + ys[3]
    correction = compute_dropout_correction(
        dropped_id=2, dropped_priv=privs[2], survivor_ids=[1, 3],
        survivor_pubs={1: pubs[1], 3: pubs[3]}, round_id=7, shape=shape,
    )
    fixed_sum = received_sum - correction
    expected_sum = xs[1] + xs[3]
    err2 = np.max(np.abs(fixed_sum - expected_sum))
    print(f"[crypto.py] erreur max apres correction de decrochage = {err2:.2e}")
    assert err2 < 1e-9, "La correction de decrochage doit restaurer exactement la somme des survivants."

    print("[crypto.py] tous les auto-tests sont passes avec succes.")
