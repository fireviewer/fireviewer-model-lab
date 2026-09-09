# Consommateurs et archives - 9 septembre 2026

Ce dépôt maintient le registre des modèles et les fonctions synthétiques
indépendantes utiles. Les deux anciennes sources sont prêtes pour archivage privé :

| Source historique | Source canonique conservée | Contrôle |
|---|---|---|
| `fireviewer/models/registry` | `registry/registry` | Quatre fichiers identiques |
| `fireviewer/models/docs` | `registry/docs` | Documents identiques |
| `fireviewer-sdg/src/fireviewer_sdg` | `src/fireviewer_model_lab/synthetic` | Helpers indépendants extraits ; imports applicatifs SDG absents |

Les pipelines Blender/Omniverse et le pack maison Hunyuan3D/Asset4Sim 001-294
sont historiques et exclus. Les six arbres CC0 Quaternius du producteur UWD sont
distincts. Aucun modèle de lancement RunPod de ces anciennes chaînes n'est
conservé comme option active.

`fireviewer-ai-worker` reste un adaptateur de compatibilité actif. Son checkout
local contient encore du travail d'entraînement non intégré ; il doit rester
préservé. Il ne constitue pas une nouvelle source canonique concurrente des
packages extraits. Avant un éventuel archivage, il faudra qualifier les lanceurs
d'entraînement et remplacer les derniers usages des anciens noms de modules.
Les changements de corpus et les campagnes GPU exigent leur propre recette ;
les tests CPU de cette migration ne les valident pas.

Le backend invoque désormais les commandes de corpus du package
`fireviewer_evidence_ingestion.mvp`, validées dans l'image CPU acceptée 0.1.1.
Les anciennes commandes restent présentes dans l'adaptateur worker pour les
consommateurs qui n'ont pas encore migré.

Les snapshots de registres conservent leurs dates et leurs notices. Il n'y a ici
aucune publication de poids, aucun lancement GPU et aucune nouvelle admission de
dataset. L'attribution technique ne remplace pas les actes à signer.
