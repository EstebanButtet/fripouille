# FRP-IA-13 — Intégration cognitive conversationnelle

## Résultat et architecture

Le flux interface → AssistantRuntime → AssistantCore → intelligence/actions
est conservé. L'assemblage partage les services d'état, rôle et perception
avec CognitiveContextProvider. Identité, mémoire et contexte social conservent
leurs modèles et providers existants. Seule la génération conversationnelle
reçoit le complément cognitif ; l'interprétation des actions ne le reçoit pas.

Les règles confirmées actives sont sélectionnées lexicalement dans la portée
de la personne résolue, puis globale : 50 candidates par portée, trois règles,
300 caractères JSON par règle et 900 au total. Leurs preuves doivent être
actives et appartenir à la même portée (20 preuves au maximum). Les règles
ne sont pas injectées sous rôle actif : le schéma v10 ne prouve pas leur rôle.
Les erreurs de lecture des règles produisent un contexte dégradé diagnostiqué.

Le complément est limité à 2 400 caractères ; le rôle à 700. Les perceptions
anonymes expirent selon FRP-IA-11. Les données sont explicitement désignées
comme faillibles et non exécutables, subordonnées à l'identité, aux permissions,
à la sécurité et à la demande actuelle. Aucun état, rôle ou règle ne vient du LLM.

Le prompt système est limité à 16 000 caractères et l'entrée à 20 000.
L'historique est omis si nécessaire, sans tronquer la demande courante ; une
demande trop grande est refusée. Le contexte Ollama conversationnel assemblé
utilise 8 192 tokens ; un budget de caractères n'est pas un comptage de tokens.
Les budgets existants de mémoire et de données sociales restent appliqués.

CognitiveTrace expose les identifiants sélectionnés, les correspondances des
règles, le rôle, l'état et la présence d'une perception. C'est une trace des
sources fournies, pas une preuve de leur influence sur le raisonnement du LLM.
Le changement d'interlocuteur efface l'historique avant l'appel suivant. Les
confirmations périmées restent soumises au contrôle de portée existant.
Le runtime refuse un second tour ou un reset concurrent, puis libère toujours
son verrou, y compris sur erreur. Les interfaces doivent passer par ce runtime.

## Validation

Tests ciblés : cognitive_integration, runtime, application, application_runtime,
model_client, learning_consolidation et people_pipeline. Les scénarios SQLite
temporaires couvrent isolation, sources invalidées, injection malveillante,
budgets, permissions, voix/texte, expiration, threads concurrents et reprise
après erreur. `scripts/validate_ia13.py --output <chemin externe>` permet un
essai explicite Ollama avec fixtures et base jetable, sans données personnelles.
Les résultats de la suite complète figurent dans le compte rendu de clôture.

## Fichiers et périmètre

application.py, core/assistant.py, cognitive_context.py, intelligence/model_client.py,
intelligence/prompt.py, intelligence/response.py, runtime.py,
interfaces/diagnostics.py, interfaces/terminal.py, learning/repository.py,
tests/test_cognitive_integration.py, scripts/validate_ia13.py et cette documentation.
Le diagnostic temporaire windows_speech.py et FRP-BIO sont exclus du jalon.
SQLite reste v10 : aucune migration ni réécriture des données ou de l'identité.

## Limites et dépendances

La pertinence des règles est lexicale, sans apprentissage implicite. Une perte
de stockage peut retirer une source du contexte ; aucune source n'est inventée.
La sincérité du texte libre du modèle et la qualité STT demandent un jalon
distinct. ROB : aucun besoin nouveau ; les capacités matérielles restent
derrière leurs validations. CAO/firmware : aucune dépendance ni modification.

## Carnet

Les contextes existants sont désormais réunis sans fusionner leurs autorités.
La provenance indique ce qui a été fourni au modèle, jamais une causalité
imaginaire. Le schéma actuel impose de laisser les règles hors des rôles actifs ;
la suite est un audit mesuré des latences et une pile vocale isolée.
