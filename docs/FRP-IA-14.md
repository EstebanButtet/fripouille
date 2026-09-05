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

## Optimisation mesurée

L'adresse par défaut est 127.0.0.1 ; l'URL reste injectable. Toutes les phases
utilisent 8192 tokens pour garder le même runner Ollama. Aucun cache cognitif,
aucune suppression d'analyse, aucun changement de schéma ou de prompt métier.
69 tests ciblés passent. Le premier tour après changement coûte encore 26,20 s
(dont 21,26 s de chargement unique) ; le suivant 3,01 s. Une seconde exécution
à chaud (`logs/ia14-after-warm.json`) donne 3,07 et 2,89 s : médiane 2,98 s,
contre 53,02 s avant. Les durées de chargement suivantes sont 1,5–2,6 ms.
Le tour hors LLM reste 9,72 ms ; contexte 3,66 ms ; social 2,39 ms ; mémoire
3,39 ms ; règles 1,05 ms ; prompt 0,024 ms. Pas de gain Python revendiqué.
Ces comparaisons utilisent les mêmes fixtures et requêtes ; le cache de préfixes
Ollama contribue au résultat à chaud. La latence vocale reste à mesurer.

## Carnet — optimisation

Deux changements de configuration retirent le principal coût observé sans
appauvrir le contexte. Le démarrage à froid reste distinct du dialogue à chaud.
La priorité suivante est la capture et la transcription locales persistantes,
sans diffusion prématurée de texte qui n'aurait pas passé les contrôles.

## Sincérité action/conversation

Le prompt contenait déjà l'interdiction de prétendre agir en conversation.
Le coeur renvoyait néanmoins le texte libre tel quel. `action_truth.py` filtre
désormais les formulations françaises courantes de faux accusés/promesses et
les demandes de tâche mal classées, dont la transcription rapportée avec Arnaud.
La réponse contrôlée annonce l'absence d'exécution et demande une action précise.
Le client Ollama vérifie la preuve lexicale des titres/contenus persistants avec
le validateur existant des candidats mémoire ; une date sans indice temporel
dans la source est rejetée. Aucun nom ou paramètre manquant n'est rempli.

Limite explicite : le filtre de texte est conservateur et lexical, pas une preuve
universelle de sincérité de toute paraphrase possible. Il peut refuser une citation
ou une paraphrase légitime ; reformuler une action explicite permet de poursuivre.
La cohérence détaillée d'une date proposée reste soumise au parseur métier existant.
Seul ActionExecutionResult constitue une preuve d'exécution ; aucun filtre de
langage ne donne de permissions ni d'accès matériel. Le cas rapporté est testé
jusqu'à l'historique, sans prénom inventé ni résultat d'action.

## Carnet — sincérité

La mauvaise transcription et la promesse hallucinée avaient deux causes distinctes.
La règle de prompt seule ne suffisait pas : une clarification applicative protège
maintenant le cas observé et les paramètres textuels non étayés. Le contrôle reste
inspectable et ses limites linguistiques sont annoncées, sans construire un agent.
