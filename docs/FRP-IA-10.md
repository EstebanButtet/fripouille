# FRP-IA-10 — Visage et expression

## Audit réel du prototype

`prototype/vector-eye` = `81fdc4c6d27d63cbdb7f8717448ad545887f0316`.
`git show --stat`, `git ls-tree` et le diff depuis l'ancêtre commun montrent
un seul changement expérimental : 258 ajouts/9 retraits dans
`firmware/fripouille_esp32/main/main.c`. C'est un contour d'œil LVGL avec deux
Bézier cubiques, trois traits néon et fallback d'allocation. Le commit indique
lui-même une validation hardware en attente. Aucun JavaScript, SVG navigateur,
webcam, suivi, pose, calibration, sept expressions ou stockage JSON n'y figure.
La source du prototype Edge décrit a été demandée ; elle n'est pas disponible
dans le dépôt inspecté. Il ne faut pas assimiler ces deux prototypes.

Le code principal possède déjà un Canvas tkinter statique et une chaîne
DisplayResponsePresenter → DisplayController → transport série cadré.
Le firmware expose PING/PONG et TEXT/OK TEXT, aucune commande expressive.

## Conception et réalisation

`ExpressiveIntent` immuable porte un symbole fermé et un motif borné.
La politique déterministe transforme état fonctionnel en neutral, focused,
curious ou concerned. Ce catalogue minimal est nouveau, dérivé des besoins
fonctionnels ; aucun catalogue de sept expressions n'a pu être réutilisé.
L'intention expire en 5 s et reset la neutralise. Ce n'est ni une émotion
mesurée, ni une identité, une mémoire, une relation ou une règle.

Le runtime produit l'intention séparément du texte. La GUI lit le contrôleur
sur son thread Tk (200 ms), affiche focused pendant le traitement et traduit
les intentions en géométrie locale. Seuls les points du contour C audité et
la couleur néon sont repris sur main ; aucune fusion ni modification firmware.
L'interface `ExpressionPresenter` prépare d'autres sorties. Aucun texte LLM
n'est analysé comme SVG, coordonnées ou commande brute.

## Validation et limites

51 tests ciblés : expressions (10), GUI, runtime, display, presentation,
windows_display et serial_transport. Compilation et diff-check réussis.
Essai avec Tk réel en fenêtre masquée : les quatre expressions créent les
trois tracés attendus ; aucune validation visuelle humaine ou sur ESP32.
SQLite v10 → v10, session uniquement, aucune dépendance externe ajoutée.

## Contrat futur ROB / firmware / CAO

IA remet un ExpressiveIntent validé. Un futur adaptateur ROB devra annoncer
les symboles réellement supportés, leur durée et un résultat d'affichage.
La présente V1 n'envoie aucune commande expressive au port COM et conserve le
chemin TEXT existant. Animation écran embarqué, calibration et validation réelle
restent à faire côté présentation/ROB ; aucun besoin de pièce CAO n'est imposé.

## Carnet

Le visage local consomme maintenant une intention distincte du texte.
L'audit a révélé que la référence Git fournie désigne un essai firmware et non
le prototype Edge décrit. Nous réutilisons seulement le contour vérifiable et
gardons le firmware intact ; la vision sociale aura une frontière indépendante.
