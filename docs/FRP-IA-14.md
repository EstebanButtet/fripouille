# FRP-IA-14 — Performance et conversation vocale locale

## Audit avant optimisation (5 septembre 2026)

Commande : `.venv/Scripts/python.exe scripts/benchmark_ia14.py --output logs/ia14-before.json --ollama`.
Python 3.14.6, Windows, RTX 3060 Ti 8 Go, qwen3.5:9b. Fixtures : 100 mémoires
et trois règles confirmées dans une base SQLite jetable. 30 mesures locales,
deux tours réels ; ces petits échantillons ne constituent pas une garantie.

| Mesure | Médiane avant |
| --- | ---: |
| Tour hors LLM | 9,70 ms |
| Complément cognitif | 3,63 ms |
| Contexte social SQLite | 2,33 ms |
| Rappel mémoire SQLite | 3,42 ms |
| Lecture règles SQLite | 1,05 ms |
| Création du prompt | 0,024 ms |
| Tour complet Ollama | 53,02 s (51,86–54,17) |
| Initialisation PowerShell + System.Speech | 219 ms |

Quatre appels Ollama séquentiels par tour mesuré : interprétation, conversation,
analyse profil, analyse mémoire. Leur ordre évite les promotions concurrentes.
L'alternance num_ctx 4096/8192 provoque des rechargements de 19,16–20,67 s,
confirmés par load_duration Ollama. Une mesure séparée de GET /api/version donne
localhost : 2059/2031/2028 ms ; 127.0.0.1 : 0,65/0,48/0,40 ms.
Le client actuel ne diffuse pas de texte partiel ; seule la réponse complète
validée est utilisable. Sérialisation et construction du prompt sont négligeables
face aux chargements réseau/modèle. Aucune justification pour un refactor massif,
un cache social ou une suppression d'analyse métier.

Les services et repositories sont déjà persistants par session. SQLite ouvre
une connexion par opération et contrôle les clés étrangères ; les quelques ms
ne justifient pas d'altérer ces garanties. Le verrou runtime refuse les tours
concurrents. Le TTS et le STT historiques relancent PowerShell à chaque opération.
La GUI possède déjà son worker ; le terminal vocal attend capture, modèle et TTS.

VRAM au début : 7238 MiB utilisés, 787 MiB libres rapportés, Ollama 100 % GPU
à 4096 tokens. Le runtime vocal sera isolé de Python 3.14, d'abord évalué sur CPU.
La mesure VRAM inclut Windows et les autres applications, pas seulement Ollama.

## Carnet — audit

L'instrumentation distingue les millisecondes Python des secondes perdues dans
les rechargements Ollama et la connexion localhost. Les lectures métier ne sont
pas le goulet principal. La décision est de stabiliser le contexte et l'adresse
locale avant d'évaluer les modèles vocaux dans un environnement séparé.
