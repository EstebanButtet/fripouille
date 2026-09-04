# Carte Codex de Fripouille

Cette carte aide à naviguer ; elle n'est pas une source d'autorité. Le code
réel reste la source de vérité. La lire avant toute recherche large et ne
mettre à jour que les entrées touchées lorsqu'une architecture évolue.

Pour une découverte pédagogique progressive du projet, lire
`docs/READING_GUIDE.md`. La présente carte reste le raccourci opérationnel.

## Entrée et orchestration

| Fichier | Rôle |
| --- | --- |
| `src/assistant_ia/__main__.py` | Sélection de l'interface terminal ou GUI. |
| `src/assistant_ia/application.py` | Assemblage de `AssistantCore` et `AssistantRuntime`. |
| `src/assistant_ia/runtime.py` | Frontière des interfaces, réponse user-facing et diagnostic du tour. |

## Conversation

| Fichier | Rôle |
| --- | --- |
| `src/assistant_ia/core/assistant.py` | Orchestration du message, des actions et des promotions mémoire/profil. |
| `src/assistant_ia/core/context.py` | Historique conversationnel en mémoire. |
| `src/assistant_ia/intelligence/turn.py` | Séparation historique / tour courant. |
| `src/assistant_ia/intelligence/model_client.py` | Client Ollama, interprétation puis génération. |
| `src/assistant_ia/intelligence/prompt.py` | Prompts, règles opérationnelles et personnalité. |
| `src/assistant_ia/interfaces/presentation.py` | Texte naturel sans plomberie technique. |

## Identité

| Fichier | Rôle |
| --- | --- |
| `src/assistant_ia/identity/models.py` | Modèle immuable d'identité stable. |
| `src/assistant_ia/identity/defaults.py` | Identité par défaut de Fripouille. |
| `src/assistant_ia/identity/context.py` | Rendu contrôlé dans le prompt. |

L'identité stable reste distincte de la mémoire, des relations, de
l'apprentissage et de l'état interne.

## Personnes

| Fichier | Rôle |
| --- | --- |
| `src/assistant_ia/people/models.py` | Modèles de personne. |
| `src/assistant_ia/people/defaults.py` | Personne par défaut. |
| `src/assistant_ia/people/context.py` | Personne active. |
| `src/assistant_ia/people/presentation.py` | Détection d'une présentation explicite. |
| `src/assistant_ia/people/person_repository.py` | Registre SQLite minimal des personnes. |
| `src/assistant_ia/people/resolution.py` | Résolution déterministe de la personne active. |
| `src/assistant_ia/people/profile_models.py` | Faits de profil confirmés et candidats distincts. |
| `src/assistant_ia/people/profile_fact_repository.py` | Persistance SQLite des faits, isolée par personne. |
| `src/assistant_ia/people/profile_promotion.py` | Doublon, correction, conflit et promotion confirmée. |
| `src/assistant_ia/people/social_models.py` | Relations bornées et observations non confirmées. |
| `src/assistant_ia/people/relationship_repository.py` | Relation conversationnelle optionnelle par personne. |
| `src/assistant_ia/people/observation_repository.py` | Observations multiples, privées et inspectables. |
| `src/assistant_ia/people/social_context.py` | Sélection sociale bornée de la personne active. |
| `src/assistant_ia/intelligence/profile_fact_candidates.py` | Analyse Ollama bornée des candidats de profil. |

## Mémoire

| Fichier | Rôle |
| --- | --- |
| `src/assistant_ia/memory/models.py` | `Memory`, candidats et liens mémoire/personne. |
| `src/assistant_ia/memory/memory_repository.py` | Souvenirs et associations SQLite inspectables. |
| `src/assistant_ia/memory/retrieval.py` | Rappel lexical borné et isolé par personne. |
| `src/assistant_ia/memory/promotion.py` | Promotion comparée dans un périmètre de sujet. |
| `src/assistant_ia/memory/subjects.py` | Attribution déterministe au locuteur actif. |
| `src/assistant_ia/intelligence/memory_candidates.py` | Analyse Ollama des candidats. |
| `src/assistant_ia/memory/repository.py` | Connexion, schéma et chemin SQLite partagé. |

## Actions et sécurité

| Fichier | Rôle |
| --- | --- |
| `src/assistant_ia/actions/action.py` | Contrat d'action et validation. |
| `src/assistant_ia/actions/registry.py` | Registre et exécution autorisée. |
| `src/assistant_ia/actions/defaults.py` | Actions locales par défaut. |
| `src/assistant_ia/security/permissions.py` | Politique de permission. |
| `src/assistant_ia/security/confirmation.py` | Confirmation des actions sensibles. |

Frontière d'autorité : le LLM propose, l'application valide. Aucun
GPIO/PWM/moteur brut ne part du LLM.

## Interfaces

| Fichier | Rôle |
| --- | --- |
| `src/assistant_ia/interfaces/terminal.py` | Interface historique et confirmations PowerShell. |
| `src/assistant_ia/interfaces/gui.py` | GUI tkinter provisoire et worker non bloquant. |
| `src/assistant_ia/interfaces/diagnostics.py` | Diagnostic console explicite. |

Terminal : `python -m assistant_ia`. GUI : `python -m assistant_ia --gui`.
Ajouter `--debug` pour les diagnostics console.

## Hardware

`src/assistant_ia/hardware/` porte la frontière PC/ESP32 (transport et
présentation matérielle). `firmware/` est hors périmètre IA sauf jalon
explicite. Préserver la version validée sur `main` et ne pas intégrer une
variante firmware expérimentale dans un commit IA.

## Tests principaux

- Orchestration : `tests/test_application.py`,
  `tests/test_application_runtime.py`, `tests/test_runtime.py`.
- Conversation : `tests/test_assistant_core.py`,
  `tests/test_model_client.py`, `tests/test_conversation_quality_prompt.py`.
- Identité : `tests/test_identity_models.py`,
  `tests/test_identity_defaults.py`, `tests/test_identity_context.py`.
- Personnes : `tests/test_people_context.py`,
  `tests/test_people_pipeline.py`, `tests/test_people_presentation.py`,
  `tests/test_person_repository.py`, `tests/test_person_resolution.py`,
  `tests/test_profile_fact_repository.py`,
  `tests/test_profile_fact_candidates.py`,
  `tests/test_profile_fact_promotion_pipeline.py`,
  `tests/test_social_repository.py`, `tests/test_social_context.py`.
- Mémoire : `tests/test_memory_repository.py`,
  `tests/test_memory_retrieval.py`,
  `tests/test_memory_promotion_pipeline.py`,
  `tests/test_memory_candidate_pipeline.py`, `tests/test_memory_people.py`,
  `tests/test_memory_person_pipeline.py`, `tests/test_memory_subjects.py`.
- Actions/sécurité : `tests/test_default_actions.py`,
  `tests/test_system_actions.py`.
- Interfaces : `tests/test_terminal.py`, `tests/test_gui.py`,
  `tests/test_response_presentation.py`.

## Pour modifier X, commencer par Y

- Comportement conversationnel : `core/assistant.py` +
  `intelligence/prompt.py` + tests associés.
- Rappel mémoire : `memory/retrieval.py` + `intelligence/model_client.py`.
- Stockage mémoire : `memory/memory_repository.py`.
- Rattachement mémoire/personne : `memory/models.py` +
  `memory/memory_repository.py` + `memory/subjects.py`.
- Interface utilisateur : `runtime.py` + `interfaces/`.
- Actions PC : `actions/` + `security/`.
- Identité stable : `identity/`.
- Faits de profil : `people/profile_models.py` +
  `people/profile_fact_repository.py` + `people/profile_promotion.py`.
- Relations et observations : `people/social_models.py` +
  `people/relationship_repository.py` + `people/observation_repository.py`.
- Contexte social du prompt : `people/social_context.py` +
  `intelligence/prompt.py` + `intelligence/model_client.py`.
