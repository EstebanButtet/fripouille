# FRP-IA-11 — Perception sociale éphémère

## Audit

Le prototype Git audité en 10 ne fournit ni capture ni détection de visage.
Windows déclare une c922 Pro Stream Webcam avec statut OK ; cette énumération
ne prouve pas un flux exploitable. Aucun moteur de vision Python n'est installé.
`CapabilityContext.visual_input` existe déjà et reste faux par défaut.
`Observation` exige une Person persistante et une source manual_entry ou
conversation_analysis ; une détection anonyme ne satisfait pas ce contrat.

## Conception et réalisation

`FaceDetection` valide les dimensions normalisées et le yaw facultatif fini.
`VisionFrame` porte un temps de capture dans l'horloge monotone locale.
`SocialVisionProvider` définit une lecture bornée et une fermeture ; le service
accepte ces trames applicatives, jamais une sortie LLM ou un nom de personne.
Il est désactivé au démarrage. start/stop sont explicites, poll fait une seule
lecture ; aucune boucle autonome. Le propriétaire du provider doit le fermer.

Le snapshot expose unavailable, present, absent ou expired, arrivée/départ et
continuité géométrique. Une piste est un entier de session alloué par le service,
jamais un identifiant persistant. La V1 ne traite qu'un visage sélectionné par
le producteur ; un déplacement >0,25 par axe ou une expiration ouvre une nouvelle
piste. Ce n'est pas une garantie que le visage appartient au même humain.
Position gauche/centre/droite ; yaw ≤20° en valeur absolue donne seulement une
orientation approximativement frontale, aucune attention ou intention mentale.

La donnée expire à 2 s. Les trames dupliquées, désordonnées, anciennes ou futures
sont refusées. Une caméra en erreur est indisponible, pas une preuve d'absence.
FRP-IA-13 peut lire le snapshot courant sans recevoir de flux de coordonnées.

## Confidentialité et persistance

Ce module ne capte ni ne stocke d'image, audio, biométrie, nom ou profil.
Seuls géométrie, temps et compteur de piste restent éphémères. Aucun historique
de trajectoire. SQLite v10 → v10, aucune migration.
Pas d'écriture automatique dans Observation, ProfileFact ou relation. Une future
observation persistée nécessitera attribution explicite par l'application,
politique de conservation et provenance visuelle dédiée, sans forger une source
conversation_analysis. Détection, tracking et résolution d'identité restent séparés.

## Validation, limites et dépendances

25 tests ciblés (vision : 13 ; social_context et social_repository), compilation,
diff-check. Données synthétiques seulement. Webcam repérée, mais aucune capture,
détection, pose ou reconnaissance d'identité testée réellement. Il manque le
provider concret du prototype Edge ; le contrat logiciel n'est pas une webcam
intégrée. Pas de nouvelle dépendance lourde installée sur cette supposition.
ROB : fournir ultérieurement les trames validées et leur durée de lecture ;
firmware : aucune modification ; CAO : aucun choix imposé. Pas de suivi moteur.

## Carnet

Le domaine visuel possède une représentation sociale bornée et anonyme.
L'Observation existante est volontairement personnelle et persistante : nous
ne lui faisons pas absorber une détection brute. La webcam est repérée mais
l'acquisition reste à intégrer ; les rôles pourront vérifier cette capacité.
