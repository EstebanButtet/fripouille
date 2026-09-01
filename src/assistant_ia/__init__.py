"""Paquet principal de l'assistant personnel local et modulaire Fripouille.

Les interfaces entrent par :mod:`assistant_ia.application` et
:mod:`assistant_ia.runtime`. La version exposée ici identifie le paquet sans
initialiser Ollama, SQLite, une interface ou le hardware lors de l'import.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
