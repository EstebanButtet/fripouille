# FRP-IA-09 — Voix locale

## Audit et conception

Python 3.14, aucune dépendance tierce installée ; terminal synchrone, GUI avec
worker. Aucun audio préexistant. Windows expose System.Speech et le moteur
MS-1036-80-DESK fr-FR ; plusieurs sorties audio sont déclarées par Windows.
Le premier backend utilise les moteurs locaux Windows, sans service cloud,
sans installation ni modification de la configuration système.

`SpeechToTextProvider.listen(cancel)` possède capture, détection de fin de
parole et transcription d'un seul énoncé ; `TextToSpeechProvider.speak` possède
synthèse et lecture. Cette frontière permet un autre moteur avec sa propre
capture sans coupler le coeur à System.Speech. La V1 est demi-duplex.

## Réalisation et utilisation

`python -m assistant_ia --voice`, puis `/listen` ; saisie texte toujours possible.
Un unique VoiceController rejoint le runtime du terminal. La transcription ne
reçoit pas davantage d'autorité que le texte ; les confirmations sensibles
restent celles du terminal. Pas de wake word, écoute continue ou retry d'action.
Silence initial 5 s ; processus STT borné à 20 s ; TTS à 120 s et 2000 caractères.
Le texte complet reste disponible. Une reconnaissance sous 0,5 est ignorée :
ce seuil de qualité du backend n'est pas une certitude sur le contenu.

`stop()` interrompt capture ou parole via Event et destruction/récupération du
processus local. Ctrl+C arrête le terminal. Pendant un appel au coeur, stop
empêche la parole suivante mais ne retire pas une action déjà engagée.
Pas d'interruption automatique par détection de voix pendant la synthèse.
Une erreur audio restitue le texte ou un message stable, sans relancer le coeur.
Les scripts PowerShell sont fixes ; tout texte est transmis comme donnée base64.

## Validation

33 tests ciblés : voice (17), terminal, runtime et main. Compilation et diff-check.
Essai réel borné : ouverture STT terminée avec silence/non-reconnaissance ; appel
TTS jusqu'à la sortie audio terminé sans erreur. Aucune phrase prononcée connue
n'a été transcrite et l'audibilité n'a pas été attestée par un humain. Ces deux
limites restent à valider avant de qualifier la conversation vocale réelle.

## Persistance, dépendances et limites

SQLite v10 → v10. Aucun enregistrement audio ni fichier temporaire ; les
transcriptions suivent ensuite les règles ordinaires de conversation/mémoire.
System.Speech/Windows PowerShell sont des composants Microsoft du système,
non redistribués ici ; pas de nouveau paquet Python, usage uniquement local.
Fallback : terminal texte. Le backend demande le moteur et une voix fr-FR.
ROB/CAO/firmware : aucune dépendance actuelle ; futur contrat audio embarqué.

Référence d'API : [Microsoft, SpeechRecognitionEngine](https://learn.microsoft.com/en-us/dotnet/api/system.speech.recognition.speechrecognitionengine?view=netframework-4.8.1).

## Carnet

Une entrée et une sortie vocales locales rejoignent maintenant le runtime texte.
Le moteur français installé permet une première intégration sans paquet externe.
Les appels audio réels ont terminé, mais intelligibilité et audibilité restent
à confirmer. Le jalon suivant séparera également l'expression de la cognition.
