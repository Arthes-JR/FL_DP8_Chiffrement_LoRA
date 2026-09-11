"""
model_build.py
================
Construction du modele "LLM + adaptateur LoRA" utilise par le prototype.

Ce module implemente, en NumPy pur (aucune dependance a torch/transformers,
non disponibles hors ligne), l'equation centrale du memoire (chapitre 2.2.4
et 6.1) :

    h = W0 x + BA x            (Eq. 2.2 du memoire)

- W0  (d_h x d_in)  : "poids de base geles", simulant une representation
                       pre-entrainee figee et identique sur tous les clients.
- A   (r x d_in)    : premiere matrice d'adaptation LoRA (rang r).
- B   (d_h x r)     : seconde matrice d'adaptation LoRA (rang r).
- w_out, b_out      : tete de classification legere, egalement entrainee
                       (represente la couche de sortie specialisee par tache,
                       p.ex. risque clinique ou fraude bancaire).

Seuls (A, B, w_out, b_out) sont entraines et transmis au serveur federe
(Chapitre 6.1.2/6.1.3 : "seuls les adaptateurs LoRA locaux sont entraines
et transmis"). W0 est generee une seule fois avec une graine partagee et
distribuee a tous les clients au demarrage : elle reste identique et gelee
partout, exactement comme le modele de base pre-entraine du memoire.

NOTE HONNETE (limite assumee, cf. chapitre 11.4.1 du memoire) :
Ce module est un SURROGATE pedagogique d'un vrai LLM. Dans un deploiement
reel, W0 proviendrait d'un modele HuggingFace (p.ex. via
`transformers.AutoModel` + `peft.LoraConfig`) et A/B seraient les matrices
LoRA injectees dans les couches d'attention/MLP du Transformer. Le protocole
federe, la confidentialite differentielle et l'agregation securisee
implementes ici s'appliquent cependant a l'identique : seule la fonction
`forward`/`backward` ci-dessous devrait etre remplacee par un appel au vrai
modele (voir README.md, section "Passage a l'echelle").
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Version numeriquement stable de la sigmoide.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


@dataclass
class ModelConfig:
    d_in: int = 20          # dimension des features d'entree (donnees tabulaires simulees)
    d_hidden: int = 32      # dimension de la representation "de base" (d_h)
    rank: int = 8           # rang r de l'adaptation LoRA (Eq. 2.2)
    seed_base: int = 42     # graine PARTAGEE pour W0 (identique sur tous les clients)
    pos_weight: float = 5.0 # <-- AJOUT : poids de la classe positive (desequilibre)


class BaseModel:
    """Represente les poids de base geles W0 (partages, jamais modifies)."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed_base)
        # Initialisation orthogonale-like (echelle 1/sqrt(d_in)) pour eviter
        # l'explosion/l'effondrement de la representation gelee.
        self.W0 = rng.normal(0.0, 1.0 / np.sqrt(cfg.d_in), size=(cfg.d_hidden, cfg.d_in))

    def project(self, X: np.ndarray) -> np.ndarray:
        """Z = X @ W0.T  (N, d_hidden) -- representation de base gelee."""
        return X @ self.W0.T


class LoRAAdapter:
    """
    Parametres entraines et echanges lors de la federation :
    A (r, d_in), B (d_hidden, r), w_out (d_hidden,), b_out (scalaire).
    """

    def __init__(self, cfg: ModelConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        rng = rng or np.random.default_rng()
        # A initialise petit-aleatoire, B initialise a zero : convention LoRA
        # standard (Hu et al., 2021) garantissant que Delta W = B A = 0 au
        # demarrage (le modele demarre identique au modele de base gele).
        self.A = rng.normal(0.0, 0.01, size=(cfg.rank, cfg.d_in))
        self.B = np.zeros((cfg.d_hidden, cfg.rank))
        self.w_out = rng.normal(0.0, 0.01, size=(cfg.d_hidden,))
        self.b_out = 0.0

    # ------------------------------------------------------------------ #
    # (De)serialisation : indispensable pour transmettre les adaptateurs
    # sur le reseau (chapitre 9, "flots de mises a jour").
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "A": self.A.copy(),
            "B": self.B.copy(),
            "w_out": self.w_out.copy(),
            "b_out": np.array([self.b_out], dtype=np.float64),
        }

    def load_dict(self, d: Dict[str, np.ndarray]) -> None:
        self.A = np.array(d["A"], dtype=np.float64)
        self.B = np.array(d["B"], dtype=np.float64)
        self.w_out = np.array(d["w_out"], dtype=np.float64)
        self.b_out = float(np.array(d["b_out"]).reshape(-1)[0])

    @staticmethod
    def zeros_like_dict(cfg: ModelConfig) -> Dict[str, np.ndarray]:
        return {
            "A": np.zeros((cfg.rank, cfg.d_in)),
            "B": np.zeros((cfg.d_hidden, cfg.rank)),
            "w_out": np.zeros((cfg.d_hidden,)),
            "b_out": np.zeros((1,)),
        }

    # ------------------------------------------------------------------ #
    # Forward / backward manuels (pas d'autograd : implementation pedagogique
    # explicite, directement tracable aux equations du memoire).
    # ------------------------------------------------------------------ #
    def forward(self, base: BaseModel, X: np.ndarray) -> Dict[str, np.ndarray]:
        Z = base.project(X)                 # (N, d_hidden)  -- gele
        U = X @ self.A.T                    # (N, r)
        V = U @ self.B.T                    # (N, d_hidden)
        H = Z + V                           # (N, d_hidden)  -- Eq. 2.2 : h = W0 x + BA x
        logits = H @ self.w_out + self.b_out  # (N,)
        P = _sigmoid(logits)
        return {"Z": Z, "U": U, "V": V, "H": H, "logits": logits, "P": P}

    def per_example_gradients(
        self, base: BaseModel, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calcule le gradient INDIVIDUEL (par exemple) de chaque parametre,
        requis pour l'ecretage DP-SGD (Algorithme 5.1 du memoire, Eq. 5.2)
        qui doit ecreter chaque gradient individuel AVANT moyenne.

        Retourne des tenseurs de forme (N, ...) : un gradient par exemple.
        """
        cache = self.forward(base, X)
        P, H, U = cache["P"], cache["H"], cache["U"]
        N = X.shape[0]

        # <-- MODIF : gradient de la BCE ponderee (pos_weight)
        # dL/dlogit = w * y * (p - 1) + (1 - y) * p
        w = self.cfg.pos_weight
        dlogit = w * y * (P - 1.0) + (1.0 - y) * P

        grad_w_out = dlogit[:, None] * H            # (N, d_hidden)
        grad_b_out = dlogit                          # (N,)

        dH = dlogit[:, None] * self.w_out[None, :]   # (N, d_hidden)
        dV = dH                                      # h = z + v => dV = dH
        grad_B = np.einsum("nh,nr->nhr", dV, U)      # (N, d_hidden, r)

        dU = dV @ self.B                             # (N, r)
        grad_A = np.einsum("nr,nd->nrd", dU, X)      # (N, r, d_in)

        return {
            "grad_A": grad_A,
            "grad_B": grad_B,
            "grad_w_out": grad_w_out,
            "grad_b_out": grad_b_out,
            "loss": self._bce_loss(P, y),
        }

    # <-- MODIF : methode (plus staticmethod) pour utiliser self.cfg.pos_weight
    def _bce_loss(self, P: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        P = np.clip(P, eps, 1 - eps)
        w = self.cfg.pos_weight
        return -(w * y * np.log(P) + (1 - y) * np.log(1 - P))

    def apply_update(self, deltas: Dict[str, np.ndarray], lr: float) -> None:
        """Applique une descente de gradient simple : theta <- theta - lr * grad."""
        self.A -= lr * deltas["grad_A"]
        self.B -= lr * deltas["grad_B"]
        self.w_out -= lr * deltas["grad_w_out"]
        self.b_out -= lr * float(deltas["grad_b_out"])

    def predict_proba(self, base: BaseModel, X: np.ndarray) -> np.ndarray:
        return self.forward(base, X)["P"]


# -------------------------------------------------------------------------- #
# Utilitaires d'agregation au niveau "dict de parametres" (utilises par
# aggregation.py mais definis ici car ils connaissent la structure du modele).
# -------------------------------------------------------------------------- #
PARAM_KEYS = ("A", "B", "w_out", "b_out")


def scale_dict(d: Dict[str, np.ndarray], factor: float) -> Dict[str, np.ndarray]:
    return {k: d[k] * factor for k in PARAM_KEYS}


def add_dicts(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: d1[k] + d2[k] for k in PARAM_KEYS}


def sub_dicts(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: d1[k] - d2[k] for k in PARAM_KEYS}


def flatten_dict(d: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(d[k]).reshape(-1) for k in PARAM_KEYS])


def unflatten_like(flat: np.ndarray, template: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    idx = 0
    for k in PARAM_KEYS:
        shape = np.asarray(template[k]).shape
        size = int(np.prod(shape)) if shape else 1
        out[k] = flat[idx: idx + size].reshape(shape)
        idx += size
    return out


if __name__ == "__main__":
    # Auto-test rapide : verifie que forward/backward tournent et que la
    # perte diminue apres quelques pas de descente de gradient locale.
    cfg = ModelConfig()
    base = BaseModel(cfg)
    rng = np.random.default_rng(0)
    adapter = LoRAAdapter(cfg, rng)

    X = rng.normal(size=(64, cfg.d_in))
    true_w = rng.normal(size=(cfg.d_in,))
    y = (X @ true_w + 0.1 * rng.normal(size=64) > 0).astype(np.float64)

    losses = []
    for step in range(200):
        grads = adapter.per_example_gradients(base, X, y)
        mean_grads = {
            "grad_A": grads["grad_A"].mean(axis=0),
            "grad_B": grads["grad_B"].mean(axis=0),
            "grad_w_out": grads["grad_w_out"].mean(axis=0),
            "grad_b_out": grads["grad_b_out"].mean(),
        }
        adapter.apply_update(mean_grads, lr=0.5)
        losses.append(grads["loss"].mean())

    print(f"[model_build.py] perte initiale={losses[0]:.4f}  perte finale={losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Le modele devrait apprendre (perte decroissante)."
    print("[model_build.py] auto-test OK.")