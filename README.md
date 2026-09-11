# Prototype -- Analyse fédérée respectueuse de la vie privée pour données sensibles 

Implémentation de référence du protocole décrit dans le mémoire : apprentissage
fédéré (FedAvg) + confidentialité différentielle (DP-SGD) + agrégation
sécurisée par masquage à secrets partagés, appliqués à un modèle de type
LoRA (adaptateurs de rang faible sur une représentation de base gelée).

## Ce que fait réellement ce code (résumé du flux)

1. **Client** (`client.py`) : entraîne le modèle **localement** (plusieurs
   époques de SGD), en **ajoutant du bruit gaussien à chaque pas de
   gradient** (DP-SGD, `privacy.py`, Algorithme 5.1 du mémoire) — le bruit
   est donc injecté **avant tout envoi**, jamais côté serveur.
2. Le client calcule ensuite sa mise à jour pondérée (`n_k/n`), puis la
   **masque** (`crypto.py`, Équation 5.4) avec des masques dérivés d'un
   échange Diffie-Hellman (X25519) avec chaque autre client — c'est
   l'équivalent, dans ce prototype, du **chiffrement du gradient** avant
   envoi : le serveur ne voit jamais une contribution individuelle en clair.
3. **Serveur** (`server.py`) : reçoit les mises à jour masquées de tous les
   clients et les **somme telles quelles** (`aggregation.py`) — les masques
   s'annulent mathématiquement dans la somme (propriété centrale de
   l'agrégation sécurisée). Le serveur obtient ainsi directement la moyenne
   pondérée FedAvg (Équation 4.2), **sans jamais déchiffrer/démasquer une
   contribution individuelle** : seule la somme finale est en clair.
4. Le serveur met à jour le **modèle global** avec cette somme, l'évalue sur
   un jeu de test tenu séparément, puis le redistribue pour le round
   suivant. Après le dernier round, le modèle global est sauvegardé en
   l'état, **prêt pour la production** (`results/global_model_final.npz`).
5. Si un client décroche en cours de round, le serveur redemande aux
   clients survivants leur **part de Shamir** de la clé privée du client
   disparu (Algorithme 5.2) pour reconstituer et retirer son masque de la
   somme, sans jamais avoir eu besoin de sa contribution en clair.

Tout ceci tourne sur de **vrais sockets TCP/TLS**, entre **vrais processus
séparés** (`subprocess`), exactement comme sur des machines distinctes.

## Structure du dépôt

```
config/base_config.yaml     Configuration centrale (hyperparamètres, réseau, sécurité)
src/
  model_build.py            Modèle LoRA (NumPy) : W0 gelé + adaptateurs A,B (Éq. 2.2)
  data_utils.py              Génération de données synthétiques + partition non-iid (Dirichlet)
  privacy.py                 DP-SGD : écrêtage + bruit gaussien (Algorithme 5.1)
  crypto.py                  ECDH, HKDF, masquage additif, partage de Shamir (Algorithme 5.2)
  aggregation.py              FedAvg pondéré + agrégations robustes (médiane, moyenne tronquée)
  metrics.py                  Précision / Rappel / F1 (Équation 11.1)
  net_protocol.py             Transport : sockets TLS, framing de messages, (dé)sérialisation NumPy
  key_generate.py             Génère certificats TLS + clés X25519 + parts de Shamir chiffrées
  server.py                   Serveur fédérateur (orchestration des rounds, agrégation, dropout)
  client.py                   Client fédéré (entraînement local DP + masquage)
  baseline_centralized.py     Baseline centralisée (config #1 du Tableau 11.1)
  run_simulation.py           Orchestrateur : génère données+clés puis lance serveur+clients
  run_experiment_suite.py     Suite 1 : 4 configurations + balayages sigma/alpha (16 exécutions)
  run_experiment_suite_2.py   Suite 2 : balayages n_clients/clip_norm/rang LoRA (14 exécutions)
  visualize.py                Génère toutes les figures du mémoire (matplotlib) à partir des CSV
requirements.txt
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate        # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

Aucune dépendance à `torch`/`transformers` : le modèle est un **surrogate
NumPy pur** du pipeline LLM+LoRA (voir "Passage à l'échelle" plus bas),
ce qui permet d'exécuter tout le pipeline sans GPU ni connexion internet.

## Lancer une simulation complète (une seule commande)

```bash
cd src
python run_simulation.py --config ../config/base_config.yaml
```

Ce script :
1. génère le jeu de données synthétique et le partitionne entre les
   `n_clients` définis dans la config (non-iid, distribution de Dirichlet) ;
2. génère les certificats TLS et les clés/parts cryptographiques
   (`key_generate.py`) ;
3. lance `server.py` puis un `client.py` par client, comme processus
   séparés, connectés en TLS sur `127.0.0.1` ;
4. affiche un résumé round par round (accuracy/précision/rappel/F1) et
   écrit `results/metrics_history.csv` + `results/global_model_final.npz`.

## Lancer manuellement (pour un vrai déploiement multi-machines)

```bash
# Sur la machine serveur (adapter network.host/port dans la config) :
python server.py --config config/base_config.yaml

# Sur CHAQUE machine cliente :
python client.py --id 1 --config config/base_config.yaml
python client.py --id 2 --config config/base_config.yaml
# ...
```

Les fichiers `keys/` (générés au préalable par `key_generate.py` puis
distribués de façon sécurisée à chaque machine) et `data/client_<id>.npz`
(le shard local de CHAQUE institution, jamais partagé) doivent être
présents sur la machine correspondante avant de lancer `client.py`.

## Reproduire les 4 configurations du Tableau 11.1 du mémoire

Changez `federation.aggregation_strategy` et `privacy.enabled` dans une
copie de `base_config.yaml` :

| # | `aggregation_strategy` | `privacy.enabled` | `security.secure_aggregation` |
|---|-------------------------|--------------------|----------------------------------|
| 1 | (utiliser `baseline_centralized.py`, pas `run_simulation.py`) | -- | -- |
| 2 | `plain_fedavg`          | `false`            | `false` |
| 3 | `plain_fedavg`          | `true`             | `false` |
| 4 | `secure_fedavg`         | `true`             | `true`  |

Pour le scénario de stress (chapitre 8.4.3, clients malveillants), utiliser
`plain_median` ou `plain_trimmed` et injecter un client corrompu (voir
`aggregation.py::__main__` pour un exemple de test unitaire correspondant).

Pour tester la reconstruction en cas de décrochage (Algorithme 5.2),
renseigner `security.simulate_dropout: {client_id: X, round: Y}`.

## Suites d'expériences automatisées (chapitre 11 + Annexe A du mémoire)

Deux scripts rejouent automatiquement TOUTES les expériences dont les
résultats figurent dans le mémoire (26 exécutions réelles au total) :

```bash
cd src
python run_experiment_suite.py     # 16 exécutions : 4 configurations + balayage sigma (6) + balayage alpha (5)
python run_experiment_suite_2.py   # 14 exécutions : balayage n_clients (4) + clip_norm (5) + rang LoRA (5)
```

Résultats écrits dans `results_suite/` et `results_suite_2/` (fichiers CSV
directement exploitables) à la racine du projet.

## Génération des figures (visualize.py)

Une fois les deux suites exécutées, génère TOUTES les figures du chapitre
11 et de l'Annexe A du mémoire (format PNG haute résolution) :

```bash
cd src
python visualize.py --suite1 ../results_suite --suite2 ../results_suite_2 --out ../figures
```

Produit `fig_convergence_curves.png`, `fig_summary_bar.png`,
`fig_privacy_sweep.png`, `fig_alpha_sweep.png`, `fig_n_clients_sweep.png`,
`fig_clip_norm_sweep.png`, `fig_rank_sweep.png` — les mêmes figures que
celles intégrées dans `rapport_TFC_theme18.pdf`, regénérables à partir de
vos propres résultats.

## Résultats déjà obtenus (exécution de validation dans cet environnement)

Avec la configuration par défaut (5 clients, alpha=0.5, 20 rounds), sur
données synthétiques de type "finance" :

| Configuration | F1 final (round 20) |
|---|---|
| Centralisé (baseline) | ~0.50 - 0.55 |
| Fédéré, sans DP ni masquage | ~0.45 |
| Fédéré + DP + agrégation sécurisée | ~0.33 |
| Fédéré + DP + agrégation robuste (médiane) | ~0.04 (voir note ci-dessous) |

**Note sur la médiane** : avec seulement 5 clients très non-iid, la
médiane coordonnée-par-coordonnée introduit un biais qui nuit fortement à
la convergence (résultat cohérent avec la littérature : l'agrégation
robuste dégrade generalement l'utilité en régime très non-iid et avec peu
de clients). C'est un résultat à part entière, à discuter au chapitre 11.4
du mémoire plutôt qu'un défaut du code.


## Passage à l'échelle : brancher un vrai LLM

Ce prototype remplace un vrai LLM par un petit modèle NumPy pour rester
exécutable sans GPU ni connexion internet (voir `model_build.py`, section
"NOTE HONNETE"). Pour un déploiement réel :

1. Remplacer `BaseModel`/`LoRAAdapter` par un modèle HuggingFace
   (`transformers.AutoModelForSequenceClassification`) enveloppé par
   `peft.LoraConfig` / `peft.get_peft_model`.
2. Remplacer `per_example_gradients` (calcul manuel) par les gradients
   PyTorch natifs (`loss.backward()`), en utilisant `torch.func.vmap` ou
   `opacus.GradSampleModule` pour obtenir des gradients **par exemple**
   (nécessaires à l'écrêtage DP-SGD).
3. Remplacer `privacy.py` par `opacus.PrivacyEngine` (comptabilité RDP
   exacte, bien plus rigoureuse que l'estimation fournie ici).
4. `crypto.py`, `net_protocol.py`, `server.py`, `client.py` restent
   valables tels quels : ils ne dépendent que de dictionnaires de tableaux
   NumPy (`to_dict()`/`load_dict()`), qu'il suffit d'adapter pour
   sérialiser les tenseurs LoRA de `peft` au lieu de `A`/`B` maison.

## Limites assumées (cohérentes avec le chapitre 11.4 du mémoire)

- Un seul thread réseau côté serveur (communications séquentielles avec
  chaque client à chaque round) : suffisant pour quelques clients, à
  paralléliser (asyncio) pour un passage à l'échelle réel.
- Le répertoire public des clés et les parts de Shamir sont pré-calculés
  hors ligne par `key_generate.py` plutôt que négociés dynamiquement par un
  relais serveur en direct (simplification documentée dans
  `key_generate.py`).
- Le masquage additif est implémenté en arithmétique flottante IEEE 754
  plutôt que sur un corps fini modulaire : suffisant pour démontrer le
  principe, pas pour une preuve de sécurité formelle.
- `privacy.estimate_epsilon` est une **approximation pédagogique** de
  l'ordre de grandeur du budget de confidentialité cumulé, pas un
  comptable de confidentialité exact (utiliser `opacus` en production).
