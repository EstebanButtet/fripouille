# FRP-IA-08 — État interne fonctionnel

## Audit et conception

Baseline : main propre à cf4d58d, 701 tests réussis (37,955 s), schéma v10.
Les recherches ciblées dans core, runtime, intelligence, identity, learning,
people, actions, security, application puis interfaces/hardware et documentation
retrouvent l'historique, les confirmations métier, les résultats structurés et
l'attente GUI. Aucun de ces objets ne constitue un état fonctionnel central.
Le coeur est le point commun texte/GUI et porte donc le service de session.

## Réalisation

`internal_state.py` expose un snapshot immuable et des événements typés.
Activité : available, engaged, waiting. Indication : blocked,
needs_strategy_change, missing_information ou aucune. Les motifs sont les
événements applicatifs, jamais une interprétation émotionnelle.
Deux échecs consécutifs de la même action demandent un changement de stratégie ;
réussite, annulation, conversation terminée, personne différente et reset
effacent la série. Une erreur de modèle signale blocked sans inventer d'action.
Le début du prochain tour conserve l'indication pour les futurs consommateurs.
Après 300 secondes hors traitement, la lecture expire l'état vers available.
Cette expiration ne supprime pas une confirmation métier dans le coeur.

Correction utilisateur et objectif terminé sont des événements d'API explicites :
aucune détection linguistique arbitraire n'est ajoutée. Pas d'urgence déduite.
FRP-IA-13 décidera du rendu cognitif. Identité et permissions sont indépendantes.

## Validation et limites

Tests : `tests.test_internal_state`, `tests.test_assistant_core`,
`tests.test_runtime`, `tests.test_application_runtime` ; compilation et diff-check.
SQLite v10 → v10 : tout est éphémère, aucune migration ni écriture supplémentaire.
Aucune dépendance logicielle externe, ROB, CAO ou firmware. Aucun essai matériel
nécessaire ; les consommateurs visage/voix pourront observer le snapshot.

## Carnet

Un état fonctionnel central suit désormais les événements réels du coeur.
L'audit a montré que l'attente GUI et les confirmations doivent conserver leurs
responsabilités propres. Nous conservons l'état uniquement en session ; la voix
pourra maintenant rejoindre le même coeur sans créer un second état métier.
