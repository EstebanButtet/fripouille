# Assistant IA personnel

Assistant IA personnel local, modulaire et sécurisé.

Le projet est destiné à fonctionner sur un ordinateur Windows puis, plus tard, à communiquer avec un robot construit autour d’un ESP32.

## Environnement

* Windows 10
* Python 3.14
* Visual Studio Code
* Windows PowerShell
* Git
* Environnement virtuel Python : `.venv`

## Architecture du projet

Le code Python utilise une architecture organisée autour du dossier `src`.

```text
assistant-ia/
├── src/
│   └── assistant_ia/
│       ├── __init__.py
│       ├── actions/
│       │   ├── __init__.py
│       │   ├── action.py
│       │   └── registry.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── assistant.py
│       │   └── context.py
│       ├── intelligence/
│       │   ├── __init__.py
│       │   ├── model_client.py
│       │   └── response.py
│       ├── interfaces/
│       │   ├── __init__.py
│       │   └── terminal.py
│       ├── memory/
│       │   ├── __init__.py
│       │   └── repository.py
│       └── security/
│           ├── __init__.py
│           ├── confirmation.py
│           └── permissions.py
├── tests/
│   └── __init__.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Modules principaux

### `core`

Coordination générale de l’assistant et construction du contexte de conversation.

### `intelligence`

Communication avec les modèles d’intelligence artificielle et gestion de leurs réponses structurées.

### `memory`

Accès aux données persistantes. L’implémentation SQLite sera ajoutée ultérieurement.

### `actions`

Définition et enregistrement des actions que l’assistant pourra proposer ou exécuter.

### `security`

Gestion des permissions, des validations et des confirmations explicites.

### `interfaces`

Interfaces utilisateur et futures interfaces matérielles, notamment le terminal et le robot.

### `tests`

Tests automatisés des composants du projet.

## Nommage du projet

Le dossier du projet et le paquet installé utilisent un tiret :

```text
assistant-ia
```

Le paquet importé dans Python utilise un underscore :

```python
import assistant_ia
```

## Reprendre le projet

Depuis une nouvelle fenêtre PowerShell :

```powershell
cd "$HOME\Dev\assistant-ia"
.\.venv\Scripts\Activate.ps1
code .
```

Le terminal doit alors commencer par :

```text
(.venv) PS C:\Users\EB\Dev\assistant-ia>
```

## Installation locale du paquet

Le projet est installé dans l’environnement virtuel en mode modifiable :

```powershell
python -m pip install --editable . --no-build-isolation
```

Cette commande n’a pas besoin d’être répétée après chaque modification du code source.

Vérifier la version installée :

```powershell
python -c "import assistant_ia; print(assistant_ia.__version__)"
```

Résultat attendu :

```text
0.1.0
```

## Principes de sécurité

* Ne jamais exécuter directement une commande produite par un modèle d’IA.
* Autoriser uniquement des actions explicitement enregistrées.
* Valider les paramètres de chaque action.
* Utiliser des listes d’applications, de fichiers et de dossiers autorisés.
* Demander une confirmation avant les actions sensibles.
* Conserver un historique des actions exécutées.
* Séparer la réflexion du modèle de l’exécution réelle des actions.

## Intégrations prévues

* Ollama pour les modèles d’IA locaux
* SQLite pour les données persistantes
* Contrôle sécurisé de certaines fonctions Windows
* Robot simulé
* Communication avec un ESP32
* Reconnaissance vocale
* Synthèse vocale

## État du développement

* Étape 1 — Préparation de l’environnement : terminée.
* Étape 2 — Création de l’architecture : terminée.
* Étape 3 — Création de l’interface terminal : prochaine étape.
