# Voix locale — FRP-IA-14

## Installation et frontières

Depuis la racine : `powershell -File scripts/setup_voice.ps1`.
Le script crée uniquement `.venv-voice` avec Python 3.10 et les versions figées
de `voice-requirements-lock.txt`, puis télécharge le modèle small multilingue.
La `.venv` Python 3.14 principale garde ses dépendances inchangées.
Modèles dans `models/whisper`, ignorés par Git. Le démarrage quotidien exige
les fichiers locaux et ne télécharge rien. Aucun audio micro n'est enregistré.

`SpeechToTextProvider` et `TextToSpeechProvider` de FRP-IA-09 sont conservés.
`VoiceProcess` échange des lignes JSON UTF-8 de 16 Ko maximum avec un worker
persistant sans socket. Le worker ne connaît ni SQLite ni le coeur. Il fournit
une transcription ou une observation VAD, jamais une autorité d'action.
Une seule requête est active ; files bornées, identifiants de réponse contrôlés,
timeout et annulation explicites. L'annulation coopérative garde les modèles ;
un calcul bloqué au-delà d'une seconde entraîne arrêt et collecte du worker.
Un prochain appel peut le recréer, sans rejouer de tour cognitif.

## Moteurs

faster-whisper 1.2.1, modèle small français multilingue, CPU int8 par défaut.
Silero VAD v6 ONNX fourni par cette version, session CPU persistante et états
récurrents remis à zéro à chaque écoute. Capture mono 16 kHz par sounddevice.
Début : probabilité ≥ 0,65 pendant 256 ms ; fin : < 0,35 pendant 704 ms.
Préambule 320 ms ; attente initiale 15 s ; parole bornée à 30 s ; texte 2000
caractères maximum. Ces seuils demandent une validation au micro réel.

System.Speech reste un repli annoncé après erreur, sans repli sur silence ou
annulation. Il redemande une phrase, car ce moteur possède sa propre capture.
Le diagnostic antérieur devient configurable : attente initiale 10 s conservée,
confiance minimale 0,5 restaurée par défaut, 0 possible explicitement en diagnostic.
Les valeurs passent comme données JSON, jamais dans du code généré.

Variables : FRIPOUILLE_VOICE_PYTHON, FRIPOUILLE_STT_MODEL,
FRIPOUILLE_STT_DEVICE, FRIPOUILLE_STT_COMPUTE, FRIPOUILLE_WHISPER_CACHE,
FRIPOUILLE_MICROPHONE (identifiant sounddevice). Modèles acceptés : tiny, base,
small, medium, large-v3, large-v3-turbo ; préparer le modèle avant changement.
Les variantes distil anglophones ne sont pas le choix français par défaut.

## GPU et sources

Le pilote CUDA n'installe pas à lui seul cuBLAS/cuDNN. faster-whisper GPU exige
les bibliothèques CUDA 12 et cuDNN 9 compatibles ; le CPU évite leur dépendance
et la concurrence VRAM avec Ollama. Aucun arrêt automatique d'Ollama par phrase.
Sources techniques : [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[Silero VAD](https://github.com/snakers4/silero-vad),
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Lancement et interruption

Texte : `.venv/Scripts/python.exe -m assistant_ia`.
Diagnostic vocal : ajouter `--voice`, puis `/listen` pour un tour ou `/voice`
pour l'écoute continue. Démarrage direct : `--continuous-voice`.
Ctrl+C en écoute continue arrête la voix et revient au terminal texte.
Les confirmations sensibles continuent à demander une saisie terminal explicite.
La réponse texte est affichée avant la synthèse ; une panne TTS la conserve.

Par défaut la capture est fermée pendant le traitement et la réponse sonore.
Le mode `--continuous-voice --barge-in` suppose un **casque isolant** : la VAD
ne sait pas distinguer le haut-parleur de l'utilisateur et aucune AEC n'est
revendiquée. Une parole soutenue arrête le TTS, garde la nouvelle phrase dans
un seul emplacement temporaire puis la transmet au runtime après le tour actif.
La synthèse est collectée avant le prochain tour. Un bruit bref ne déclenche
pas l'événement de début ; les seuils/hystérésis sont ceux décrits ci-dessus.
Le repli System.Speech n'offre pas ce mode : revenir à l'écoute sans barge-in.

## Carnet — conversation continue

Le terminal peut maintenant enchaîner les tours sans commande répétée, afficher
le texte avant le son et revenir au texte sur arrêt. L'interruption repose sur
une observation VAD soutenue et ne relance aucune action déjà engagée. Le mode
avec casque est explicite ; l'écho sur haut-parleur reste un sujet distinct à valider.

## Carnet — providers et STT

Les dépendances ML vivent maintenant dans un worker remplaçable et persistant.
Une transcription reste du texte non fiable transmis aux contrôles habituels.
Le CPU est retenu pour préserver la VRAM ; la prochaine validation compare
les transcriptions et latences avec les mêmes phrases au microphone.

Essai local effectué : chargement du vrai worker small CPU int8 en 1,68 s,
ouverture du microphone Logitech, annulation après 2 s et processus conservé.
Un blocage d'import NumPy depuis un thread secondaire sous Python 3.10 Windows
a été corrigé en initialisant les DLL sur le thread principal avant la boucle IPC.
Ce test vérifie capture/arrêt, pas encore la compréhension de la voix humaine.
