# Guide de lecture du code de Fripouille

Ce guide accompagne une personne qui connaît les bases de Python mais découvre
les dataclasses, les protocoles, les repositories et une architecture en
couches. Il ne décrit pas un projet futur : il suit le code présent après
FRP-IA-03, FRP-IA-04, FRP-IA-05 et FRP-IA-02B.

La meilleure méthode n'est pas de lire les fichiers par ordre alphabétique.
Suivez d'abord un message de l'interface jusqu'à sa réponse, puis ouvrez les
domaines spécialisés.

## Quand l'utilisateur envoie un message, que se passe-t-il ?

Dans le chemin terminal historique :

```text
python -m assistant_ia
        |
        v
interfaces/terminal.py          lit le texte
        |
        v
AssistantRuntime                sépare réponse visible et diagnostic
        |
        v
AssistantCore                   orchestre le tour
        |
        +--> OllamaModelClient  propose une intention validée
        |
        +--> ActionRegistry     exécute une action autorisée
        |        `--> repositories SQLite / lanceur Windows
        |
        `--> mémoire            analyse un candidat puis demande confirmation
        |
        v
interfaces/presentation.py      retire la plomberie technique
        |
        v
terminal ou GUI                 affiche la réponse
```

Les chemins « action » et « conversation » sont exclusifs pour un tour. Ollama
peut proposer une intention, mais l'application garde l'autorité : elle valide
le nom, les paramètres, la permission et, si nécessaire, le consentement.

Pour une conversation ordinaire, le client peut rappeler quelques souvenirs
pertinents après avoir établi qu'il ne s'agit pas d'une action. Ces souvenirs
sont du contexte non autoritatif : ils ne deviennent jamais des ordres.

## Niveau 1 — suivre le trajet d'un message

Lisez ces fichiers dans cet ordre :

1. `src/assistant_ia/__main__.py`
   choisit le terminal historique ou la GUI provisoire.
2. `src/assistant_ia/interfaces/terminal.py`
   montre une boucle d'interface simple et la confirmation interactive.
3. `src/assistant_ia/application.py`
   assemble toutes les dépendances concrètes de l'application.
4. `src/assistant_ia/runtime.py`
   constitue la frontière commune des interfaces.
5. `src/assistant_ia/core/context.py`
   conserve l'historique temporaire de la conversation.
6. `src/assistant_ia/core/assistant.py`
   orchestre réellement un tour, une action et la promotion mémoire.
7. `src/assistant_ia/interfaces/presentation.py`
   transforme le résultat interne en réponse naturelle.

À la fin de ce niveau, vous devez pouvoir répondre à trois questions : qui
possède l'historique, qui décide d'exécuter une action, et où le texte visible
est-il construit ?

### Runtime

`AssistantRuntime` est un adaptateur entre une interface et le coeur. Il appelle
le coeur, photographie les diagnostics, prépare le texte visible et appelle un
presenter facultatif. Il n'interprète pas lui-même le message.

Le même runtime doit vivre pendant toute la session : il possède le même coeur,
qui possède le même contexte conversationnel.

### Core

`AssistantCore` est l'orchestrateur métier. Il enchaîne des composants sans
absorber leurs responsabilités :

- le client de modèle interprète et génère ;
- le registre valide et exécute les actions disponibles ;
- le contexte ordonne les messages temporaires ;
- l'analyseur propose des candidats mémoire ;
- le service de promotion compare et applique une proposition confirmée.

Le coeur ne connaît ni `input`, ni tkinter, ni les détails SQL ou Win32.

### Presenter

Un presenter est un objet doté d'une méthode `present(response)`. Le `Protocol`
`ResponsePresenter` décrit cette forme sans imposer de classe mère. L'écran
physique peut donc recevoir la même réponse finale qu'une autre interface,
alors que le coeur ignore complètement son existence.

## Niveau 2 — comprendre l'intelligence et Ollama

Lisez ensuite :

1. `src/assistant_ia/intelligence/turn.py`
   sépare le tour courant d'un historique récent et borné.
2. `src/assistant_ia/intelligence/intent.py`
   définit la liste fermée des intentions et de leurs paramètres.
3. `src/assistant_ia/intelligence/interpretation.py`
   regroupe le résultat validé de l'interprétation.
4. `src/assistant_ia/intelligence/response.py`
   décrit ce que le client remet au coeur.
5. `src/assistant_ia/intelligence/model_client.py`
   effectue les appels HTTP locaux à Ollama.
6. `src/assistant_ia/intelligence/prompt.py`
   assemble les prompts à partir des contextes validés.
7. `src/assistant_ia/intelligence/conversation.py`
   valide les modes de génération conversationnelle spécialisés.
8. `src/assistant_ia/intelligence/allocation.py`
   vérifie exactement les répartitions possédant un total fixe.

### ConversationContext et ConversationTurn

`ConversationContext` possède tous les messages temporaires de la session.
`ConversationTurn` est une vue préparée pour un seul appel :

- `history` contient un suffixe récent, dans ses limites de taille ;
- `current_user_message` contient la demande qui doit être interprétée.

Cette séparation empêche de prendre un ancien message pour l'ordre actuel et
permet de borner l'entrée d'Ollama sans couper une phrase en plein milieu.

### Intent

Un `Intent` n'est pas une action exécutée. C'est un nom autorisé et un mapping
immuable de paramètres textuels. Sa validation garantit une forme correcte ;
`Action` vérifie ensuite les paramètres obligatoires et facultatifs ; le handler
applique enfin les règles métier propres à l'opération.

### Validation déterministe

« Déterministe » signifie qu'à entrée identique, le code Python applique les
mêmes règles et obtient le même résultat. Ollama est utile pour proposer une
interprétation, mais les décisions sensibles ne reposent pas uniquement sur sa
bonne volonté : schémas JSON, listes fermées, preuves textuelles et calculs
`Decimal` sont revérifiés localement.

## Niveau 3 — comprendre la mémoire

Ordre conseillé :

1. `src/assistant_ia/memory/models.py`
   distingue tâche, souvenir, candidat et entrée de journal.
2. `src/assistant_ia/memory/repository.py`
   ouvre SQLite, gère les transactions et les migrations.
3. `src/assistant_ia/memory/memory_repository.py`
   traduit les opérations de souvenirs en SQL.
4. `src/assistant_ia/intelligence/memory_candidates.py`
   extrait puis filtre des candidats non persistants.
5. `src/assistant_ia/memory/promotion.py`
   compare un candidat aux souvenirs existants.
6. `src/assistant_ia/memory/retrieval.py`
   classe les souvenirs utiles à une conversation.
7. `src/assistant_ia/memory/task_repository.py`
   illustre un autre repository et une transition d'état.
8. `src/assistant_ia/memory/journal_repository.py`
   distingue date métier et instant d'enregistrement.

### Repository

Un repository est la couche qui traduit les opérations métier en lectures et
écritures SQLite. Les couches supérieures demandent par exemple
`save_candidate(candidate)` ; elles ne construisent pas de requête SQL.

Le bloc `with database.connect()` délimite une transaction. Une sortie normale
valide les changements ; une exception les annule avant la fermeture de la
connexion. Les `?` dans le SQL reçoivent des paramètres séparés et évitent de
concaténer directement un texte utilisateur à la requête.

### MemoryCandidate

Un `MemoryCandidate` est une information personnelle possible, validée mais
encore absente de SQLite. Ses champs ont un sens précis :

- `content` : formulation concise proposée pour le futur souvenir ;
- `source_text` : extrait exact du message utilisateur qui sert de preuve ;
- `confidence` : confiance dans la fidélité de l'extraction, pas probabilité
  que l'affirmation soit vraie dans le monde.

L'analyseur ne donne ni identifiant au candidat ni permission de l'enregistrer.

### MemoryPromotionProposal

Le service de promotion compare un candidat aux souvenirs persistants et
produit une `MemoryPromotionProposal` : création, déjà connu, doublon possible,
mise à jour ou conflit. La proposition n'écrit rien.

`AssistantCore` demande le consentement lorsque nécessaire. Après un « oui »,
le service recalcule la proposition pour vérifier qu'elle est encore actuelle,
puis seulement le repository écrit dans SQLite.

### Rappel contextuel

`ContextualMemoryRetriever` effectue un classement lexical local et
inspectable. `RetrievedMemory.score` mesure le recouvrement des termes de la
requête, pas la vérité ou l'importance universelle du souvenir. Le rappel est
en lecture seule et les budgets du prompt limitent le nombre et la taille des
souvenirs injectés.

## Niveau 4 — comprendre l'apprentissage comportemental

Ordre conseillé :

1. `src/assistant_ia/learning/models.py`
2. `src/assistant_ia/learning/repository.py`
3. `src/assistant_ia/learning/service.py`

Une `BehavioralExperience` n'est ni un souvenir ni une observation sociale :
elle décrit une stratégie tentée dans un contexte et son résultat. Sa
provenance est structurée et sa portée est soit globale, soit celle d'une
personne persistante déjà résolue par l'application.

Une `BehavioralLessonCandidate` cite au moins une expérience source. Même
ainsi, elle ne constitue jamais une règle confirmée. FRP-IA-05 ne l'injecte
pas dans les prompts et ne la crée pas automatiquement ; FRP-IA-06 et 07
porteront respectivement l'évaluation approfondie et la consolidation.

## Niveau 5 — comprendre les actions et la sécurité

Lisez :

1. `src/assistant_ia/actions/action.py`
2. `src/assistant_ia/actions/registry.py`
3. `src/assistant_ia/actions/defaults.py`
4. `src/assistant_ia/security/permissions.py`
5. `src/assistant_ia/security/confirmation.py`
6. `src/assistant_ia/system/windows.py`
7. `src/assistant_ia/capabilities/context.py`

Le trajet d'une demande système est volontairement long :

```text
JSON Ollama
   -> Intent validé
   -> ActionRegistry (nom enregistré ?)
   -> Action (paramètres exacts ?)
   -> PermissionPolicy
   -> confirmation de l'interface si nécessaire
   -> WindowsApplicationLauncher (cible dans la liste blanche ?)
   -> subprocess sans shell
```

`CapabilityContext` décrit au modèle ce qui est réellement enregistré dans
l'application. Connaître un nom d'intention ne suffit pas à rendre sa capacité
disponible. Les capacités futures restent annoncées comme absentes.

## Niveau 6 — comprendre identité et personne active

Lisez :

1. `src/assistant_ia/identity/models.py`
2. `src/assistant_ia/identity/defaults.py`
3. `src/assistant_ia/identity/context.py`
4. `src/assistant_ia/people/models.py`
5. `src/assistant_ia/people/context.py`
6. `src/assistant_ia/people/presentation.py`

`AssistantIdentity` est la configuration stable de Fripouille. Sa dataclass est
`frozen=True` : ses champs ne peuvent pas être réassignés après construction.
Une réponse du modèle ne la fait jamais évoluer.

`ActivePersonContext` répond à une question beaucoup plus petite : « qui parle
dans cette session ? ». Une présentation explicite peut changer ce nom jusqu'au
prochain reset. Les profils, relations et observations de FRP-IA-04 sont
persistés dans des modèles séparés et ne modifient pas l'identité.

## Niveau 7 — comprendre les interfaces et le hardware

Pour la GUI :

1. `src/assistant_ia/interfaces/gui.py`
2. `src/assistant_ia/interfaces/diagnostics.py`

tkinter doit rester réactif. La fenêtre lance donc `runtime.process_message`
dans un worker. Le worker ne modifie aucun widget : `root.after` remet la
réponse dans la boucle graphique principale. Le contrôleur séparé peut être
testé sans écran réel.

Pour l'écran physique :

1. `src/assistant_ia/hardware/transport.py`
2. `src/assistant_ia/hardware/display.py`
3. `src/assistant_ia/hardware/presentation.py`
4. `src/assistant_ia/hardware/serial_transport.py`
5. `src/assistant_ia/hardware/windows_serial.py`
6. `src/assistant_ia/hardware/windows_display.py`

### Frontière hardware

La règle fondamentale est :

```text
LLM
  -> intention proposée
  -> validations, registre, permissions et confirmations applicatives
  -> contrôleur logiciel de haut niveau
  -> transport cadré
  -> connexion Windows
  -> hardware
```

Le LLM ne commande jamais directement GPIO, PWM ou moteur. Le transport ne
comprend pas l'intention conversationnelle ; il échange seulement une commande
déjà construite par une couche contrôlée. Le firmware ESP32 est un autre côté
de cette frontière et n'est pas à modifier pendant une passe documentaire IA.

## Concepts Python rencontrés dans ce dépôt

### Dataclass immuable

`@dataclass(frozen=True, slots=True)` génère notamment le constructeur et la
comparaison, interdit les réassignations ordinaires et limite les attributs aux
champs déclarés. `__post_init__` valide juste après le constructeur ; le code y
utilise `object.__setattr__` uniquement pour enregistrer une forme normalisée
avant que l'objet ne soit remis au reste de l'application.

### Protocol

Un `Protocol` décrit les méthodes attendues sans fournir l'implémentation. Un
faux client Ollama, une fausse connexion série ou un presenter de test peuvent
donc être injectés s'ils possèdent la bonne méthode.

### Injection de dépendances

Au lieu de construire SQLite ou Ollama au milieu d'une méthode métier,
`application.py` crée les objets et les passe aux constructeurs. Cela rend les
frontières visibles et permet aux tests de remplacer un service externe par un
double déterministe.

### Callback

Un callback est une fonction passée comme valeur pour être appelée plus tard.
Le handler de confirmation, l'horloge injectable et le reporter d'erreur GUI
en sont des exemples. Leur type `Callable[...]` précise les arguments et la
valeur de retour attendus.

### Modèle immuable et copie défensive

Les tuples, `frozenset` et `MappingProxyType` empêchent une modification
accidentelle après validation. Quand un contexte interne utilise une liste
mutable, sa propriété publique retourne un tuple pour ne pas exposer cette
liste directement.

## Repères sur ce qui n'est pas encore présent

La documentation du code actuel ne doit pas transformer la feuille de route en
fonctionnalités existantes :

- FRP-IA-04 profils et relations : terminé ;
- FRP-IA-05 fondations de l'apprentissage comportemental : terminé ;
- FRP-IA-06 retour d'expérience : futur ;
- FRP-IA-07 consolidation : futur ;
- FRP-IA-08 état interne : futur ;
- FRP-IA-09 voix : futur ;
- FRP-IA-10 vrai visage : futur ;
- FRP-IA-11 vision sociale : futur.

Le terminal reste l'interface historique et la GUI de FRP-IA-02B une interface
provisoire. La mémoire contextuelle documentée ici correspond à FRP-IA-03.

## Comment vérifier sa compréhension

Après chaque niveau, choisissez un test ciblé indiqué dans `docs/CODEX_MAP.md`
et comparez son scénario au module de production. Les tests illustrent les
contrats, mais le code de `src/assistant_ia/` reste la source de vérité.

Pour un premier exercice sans modifier le projet, suivez mentalement ces trois
messages :

1. « Salut, je m'appelle Alice » : personne active puis conversation Ollama.
2. « Crée une tâche acheter du pain » : intention, registre, repository SQLite.
3. Une préférence personnelle stable : conversation, candidat non persistant,
   proposition, réponse oui/non, puis éventuelle écriture.

Si vous savez nommer le module responsable à chaque flèche, vous possédez déjà
la carte essentielle de Fripouille.
