# GRO720 - Réseaux de Neurones Artificiels

## Pré-requis

[uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2) n'est pas nécessaire pour exécuter les scripts, mais simplifie la gestion des environnements virtuels Python.

## Comment utiliser

Windows Terminal (Powershell):

```pwsh
cd path\to\gro720
uv sync  # Reproduire l'environnement virtuel (seulement si inexistant)
.venv\Scripts\Activate.ps1  # Activer l'environnement virtuel
```

> Alternative sans `uv`:

```pwsh
cd path\to\gro720
python -m venv .venv
.venv\Scripts\Activate.ps1
pip3 install numpy matplotlib tqdm  # NOTE: the versions MAY differ from the `uv` method
```

Pour exécuter un script, utiliser l'une des commandes suivantes:

```pwsh
uv run laboratoires/l1/e1.py
uv run -m problematique.train_mnist
python laboratoires/l1/e1.py
python -m problematique.train_mnist
```

> _N.B._ Les commandes `python` requièrent l'envrionnement virtuel activé.

UNIX:

```bash
cd path\to\gro720
uv sync  # Reproduire l'environnement virtuel (seulement si inexistant)
source .venv/bin/activate  # Activer l'environnement virtuel
```

Pour les commandes Python, remplacer `python` par `python3` (ou votre alias vers l'exécutable Python).

## Mise à jour

Pour mettre à jour les versions des dépendances, utiliser `uv`.

```pwsh
uv self update  # Mettre à jour uv
uv sync --upgrade  # Mettre à jour les dépendances
uv lock --upgrade  # Mettre à jour les requis d'environnement (pour distribution)
```
