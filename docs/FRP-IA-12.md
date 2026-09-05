# FRP-IA-12 — Rôles fonctionnels temporaires

## Audit et conception

`AssistantIdentity.role` décrit l'identité stable, et ConversationMessage.role
le locuteur du message : aucun n'est un métier activable. Le registre d'actions
expose ses capacités réelles ; aucun contrôle robotique, prise vidéo ou suivi
de cible n'est disponible. Le schéma d'apprentissage v10 ne porte pas de rôle.
Une V1 à zéro ou un rôle actif suffit ; défaut = fonctionnement habituel.

## Réalisation

`FunctionalRole` immuable : identifiant, nom, objectif, capacités requises,
quatre priorités/contraintes au maximum par catégorie. Catalogue applicatif
en lecture seule, pas de chargement de définition depuis le LLM.
`RoleService` valide activation/remplacement contre une fonction de capacités.
Un refus conserve le rôle précédent ; perte de capacité, personne différente
ou reset le désactivent. Pas de persistance de l'activation.

Guide exige conversation et est disponible. Observateur exige social_vision ;
cameraman exige record_video et follow_target. Ces deux derniers restent refusés
dans l'assemblage actuel. Aucun peintre fictivement doté d'un bras n'est ajouté.
Les rôles ne touchent pas aux permissions, à l'identité ou au registre.

Demandes utilisateur complètes traitées avant le modèle dans le coeur commun :
`Passe en mode guide.`, `/role guide`, `/role off`, `Passe en mode assistant.`.
Une citation ou demande ambiguë ne déclenche pas l'activation. Les confirmations
mémoire/profil existantes conservent leur priorité. Le modèle n'active aucun rôle.
Le rôle est exposé via core.roles ; son injection bornée arrive en FRP-IA-13.

## Apprentissage et SQLite

SQLite v10 → v10. Une BehavioralAttempt peut porter un role_id éphémère.
Le service assemblé refuse de persister une expérience pendant un rôle actif,
ou une tentative portant un rôle même après désactivation. Il refuse également
la confirmation d'une règle pendant un rôle. Les sources et règles historiques
restent intactes ; leur rôle n'est pas inventé. L'apprentissage métier persistant
attend un vrai besoin et une migration de portée dédiée. Les lectures restent
possibles, l'apprentissage habituel reste fonctionnel, sans boucle automatique.

## Validation et limites

53 tests ciblés : roles (15), learning_service, learning_consolidation,
application et assistant_core. Compilation et diff-check réussis.
Aucune dépendance nouvelle, ni essai matériel requis pour le rôle guide.
ROB devra annoncer des capacités de prise de vue et suivi haut niveau avant
cameraman. CAO/firmware : aucun changement ; aucune commande moteur.

## Carnet

Un rôle temporaire peut maintenant être demandé au même coeur depuis chaque
interface. L'audit confirme que les rôles descriptifs existants ne convenaient
pas à une activation métier. La V1 bloque les apprentissages sans portée de rôle
persistante ; FRP-IA-13 assemblera ces indications sans élargir les permissions.
