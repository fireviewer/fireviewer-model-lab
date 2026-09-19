# fireviewer-model-lab

## Repères documentaires — 19 septembre 2026

- **Rôle :** Préparation des corpus, recettes d’entraînement, évaluations, benchmarks et registre des modèles.
- **Statut :** Actif — package v0.1.1. Les anciens dépôts `models` et `fireviewer-sdg` restent archivés.
- **Entrées :** Corpus vérifiés, manifestes de droits, configurations et checkpoints.
- **Sorties :** Recettes, registres, rapports d’évaluation, artefacts explicitement synthétiques et métadonnées de modèles.
- **Limites :** Ne pas comparer des métriques issues de protocoles différents. Conserver split, révision, corpus, environnement et limites de chaque score.

[Fiche du dépôt](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/repositories/fireviewer-model-lab.md) · [Architecture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/ARCHITECTURE.md) · [Statuts et vocabulaire](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/STATUTS_ET_VOCABULAIRE.md).

Cette revue documentaire ne renouvelle aucun test ni aucune réception. Les procédures, versions et preuves techniques ci-dessous conservent leur périmètre et leur date.

Le [catalogue public Hugging Face](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/HUGGINGFACE.md) distingue les benchmarks indépendants de D-FINE/RT-DETR/YOLO, la validation d’entraînement RF-DETR et le pilote DINOv3. Les corpus privés gardent leurs cards et droits propres.

> **Source active FV · public.** Préparation des corpus, entraînement, benchmarks et registre des modèles. Voir [où travailler, quoi commiter et comment reprendre](ORGANISATION.md).

Model recipes, corpus preparation, evaluation and explicitly synthetic fixtures.

Python package: `fireviewer_model_lab`. Version: `0.1.1`.

## Installation

Install the versioned release wheels (including versioned FireViewer dependencies) from the release bundle. No sibling source checkout is required.

```sh
python -m pip install --find-links /path/to/release/wheels fireviewer-model-lab==0.1.1
python -m pytest tests -q
```

Optional model/provider environments are separate extras and retain their existing upstream constraints. Model weights, credentials, datasets and local evidence are external inputs.

## Ownership and compatibility

Target account: `fireviewer`. Target stewardship: Association FIRE-VIEWER. Historical authorship and AGPL-3.0-or-later notices are retained. This technical extraction is not a signed assignment of rights.

Source correspondence and hashes are recorded in the migration dossier. Existing schema IDs, algorithm revisions and evidence/publication gates are preserved. The former `firewarning_worker` or backend module paths are compatibility adapters in their original repository.

## Delivery boundary

Docker was deferred during the initial source delivery. The resumed private container phase, pinned images and acceptance limits are documented in [fireviewer-docker](https://github.com/fireviewer/fireviewer-docker). Production deployment remains separate. CPU/schema tests do not qualify GPU, visual or scientific performance.

## Sources et commandes propres au composant

Le registre maintenu se trouve dans [`registry/registry`](registry/registry) et
sa documentation dans [`registry/docs`](registry/docs). Les anciennes sources
`fireviewer/models` et `fireviewer-sdg` sont conservées en archives restaurables hors des dépôts actifs ; voir
[les consommateurs et critères de retrait](docs/LEGACY-CONSUMERS.md).

Commande locale de QA : `fireviewer-model-qa --help`. Recettes AI dans `training`, recettes locales supplémentaires dans `recipes`, registres documentaires dans `registry`. Le bundle de composition 0.1.0 est explicitement re-scellé ; les anciens pins et la divergence du snapshot sont conservés dans `registry/legacy-compose-pins.json`. Aucun lancement cloud n’a été exécuté.

Les dépendances de base sont verrouillées avec hashes dans `requirements.lock.txt` (Python 3.13). Installer les wheels versionnés du même bundle via `--find-links`. Les extras lourds restent liés à leurs versions existantes et ne qualifient aucun GPU. Les tests de composant et leurs dépendances de test sont recensés dans le dossier unique de migration.

## Ouverture du code source — 19 septembre 2026

Ce dépôt fait partie du premier lot de huit composants FIRE-VIEWER ouvert au public sur décision du mainteneur. Le code original reste sous **AGPL-3.0-or-later** et la documentation originale sous **CC BY 4.0**, avec les notices et droits tiers existants.

Cette ouverture porte sur le code, son historique et les artefacts de développement déjà associés au dépôt. Les services déployés, comptes, données, corpus, modèles, secrets et autorisations des ressources externes gardent leur propre périmètre. Les sources des sites, du backend, des applications Android et de l’infrastructure restent privées. La visibilité publique ne constitue ni une nouvelle recette fonctionnelle ni un acte de cession des droits.

[Inventaire et périmètre d’ouverture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/OPEN_SOURCE.md).
