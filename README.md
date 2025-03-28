# Modèle de Stock Picking

Ce projet vise à développer un modèle de stock picking permettant de sélectionner des actions à fort potentiel de performance en combinant une analyse fondamentale et factorielle.

## Objectifs

- Concevoir un algorithme de sélection d'actions basé sur des critères financiers et de valorisation
- Utiliser des facteurs fondamentaux (P/E, ROE, ROIC, croissance des revenus, etc.)
- Intégrer un modèle quantitatif pour l'analyse des données historiques
- Ajouter une couche qualitative (gouvernance, stratégie, potentiel de marché)
- Automatiser la collecte et l'analyse des données

## Structure du projet

- `/data`: Données brutes et traitées
- `/src`: Code source du modèle
  - `/src/data`: Scripts de collecte et traitement des données
  - `/src/models`: Implémentation des modèles de scoring et d'analyse
  - `/src/visualization`: Outils de visualisation et dashboard
- `/notebooks`: Jupyter notebooks pour l'exploration et l'analyse
- `/tests`: Tests unitaires et d'intégration
- `/docs`: Documentation du projet

## Installation

```bash
# Cloner le répertoire
git clone https://github.com/Kyac99/stock-picking-model.git
cd stock-picking-model

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

Instructions détaillées à venir

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.