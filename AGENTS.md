# Fripouille — repères permanents pour Codex

## Projet

- Fripouille est un assistant IA incarné dans un robot.
- Environnement cible : Windows 10, Python et Ollama avec `qwen3.5:9b`.
- Paquet principal : `src/assistant_ia`.
- Lancement historique : `python -m assistant_ia` (interface terminal).
- Préserver la nomenclature des jalons `FRP-IA` dans la documentation,
  les tests et les commits.

## Avant toute recherche

Lire `docs/CODEX_MAP.md` avant de rechercher largement dans le dépôt.
La carte indique les premiers fichiers à ouvrir selon le domaine.
Le code réel reste la source de vérité : si une modification rend une entrée
obsolète, mettre à jour uniquement cette entrée.

Inspecter localement les fichiers concernés et leurs dépendances directes avant
toute modification. Utiliser `rg` par chemin ou symbole. Ne pas parcourir tout
`src/`, tous les tests ou `firmware/` sans contradiction concrète ou échec
de test qui le justifie.

Ne pas reconstruire une fonction déjà présente : retrouver et réutiliser
l'implémentation existante.

## Architecture et autorité

Le flux principal est :

`interface -> AssistantRuntime -> AssistantCore -> intelligence / actions`

- Le LLM propose ; l'application valide.
- Le LLM ne commande jamais directement un GPIO, un PWM ou un moteur brut.
- Les actions passent par les registres, validations, permissions et
  confirmations applicatives.
- L'identité stable est séparée de la mémoire, des relations, de
  l'apprentissage et de l'état interne.
- Ne pas faire évoluer `AssistantIdentity` depuis une réponse du modèle.
- Le terminal reste l'interface historique et un outil de diagnostic.

## Tests

Toute modification de comportement doit ajouter ou adapter des tests ciblés.

Commandes usuelles depuis la racine, environnement virtuel activé :

```powershell
python -m unittest tests.test_nom_du_module
python -m unittest discover -s tests
git diff --check
git status --short
```

Préférer les tests unitaires sans service Ollama, matériel ou écran réel.

## Limites du dépôt

`firmware/fripouille_esp32/main/main.c` contient actuellement une modification
locale hors des jalons IA. Ne jamais l'inclure accidentellement dans un diff,
un index ou un commit IA. Ne modifier ni le hardware ni le firmware sans jalon
explicite qui les place dans le périmètre.

Avant tout commit : inspecter `git diff --cached`, vérifier l'absence de base
SQLite ou fichier temporaire, et confirmer que `main.c` n'est pas indexé.
