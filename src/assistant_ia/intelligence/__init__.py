"""Couche d'intégration du modèle de langage et de validation de ses sorties.

Les modules de ce paquet construisent les prompts, appellent Ollama, puis
transforment ses propositions non fiables en objets Python strictement
validés. Ils n'exécutent ni action système, ni écriture en mémoire persistante.
"""
