"""Frontière logicielle entre l'application PC et le hardware de Fripouille.

Le chemin d'autorité doit toujours rester : LLM -> intention proposée ->
validation applicative -> contrôleur de haut niveau -> transport -> matériel.
Le modèle ne commande jamais directement un GPIO, un PWM ou un moteur brut.
"""
