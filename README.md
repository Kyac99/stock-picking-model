# Modèle de Stock Picking

Ce projet développe un modèle de stock picking permettant de sélectionner des actions à fort potentiel de performance en combinant analyse fondamentale, technique et factorielle.

## Objectifs

- Concevoir un algorithme de sélection d'actions basé sur des critères financiers et de valorisation
- Utiliser des facteurs fondamentaux (P/E, ROE, ROIC, croissance des revenus, etc.)
- Intégrer un modèle quantitatif pour l'analyse des données historiques
- Ajouter une couche qualitative (gouvernance, stratégie, potentiel de marché)
- Automatiser la collecte et l'analyse des données
- Optimiser des portefeuilles basés sur les scores du modèle

## Structure du projet

- `/data`: Données brutes et traitées
- `/src`: Code source du modèle
  - `/src/data`: Scripts de collecte et traitement des données
  - `/src/models`: Implémentation des modèles de scoring et d'optimisation
  - `/src/visualization`: Outils de visualisation et dashboard
- `/notebooks`: Jupyter notebooks pour l'exploration et l'analyse
- `/tests`: Tests unitaires et d'intégration
- `/docs`: Documentation du projet

## Fonctionnalités principales

1. **Collecte de données financières**
   - Données de prix historiques via Yahoo Finance
   - Données fondamentales (états financiers, ratios, etc.)
   - Support optionnel pour Alpha Vantage

2. **Analyse multifactorielle**
   - Analyse fondamentale : P/E, ROE, ROIC, croissance, marges...
   - Analyse technique : tendance, momentum, volatilité...
   - Scoring combiné pour identifier les meilleures opportunités

3. **Optimisation de portefeuille**
   - Méthode de Markowitz (Mean-Variance)
   - Optimisation basée sur les scores
   - Optimisation par parité de risque
   - Backtest des portefeuilles optimisés

4. **Visualisation et dashboard**
   - Graphiques interactifs pour l'analyse des résultats
   - Dashboard pour le suivi des performances

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

### Configuration

Le modèle utilise un fichier `config.ini` pour ses paramètres principaux. Vous pouvez éditer ce fichier pour :

- Spécifier les tickers à analyser
- Configurer les pondérations des facteurs
- Définir les paramètres d'optimisation de portefeuille
- Ajouter votre clé API Alpha Vantage (optionnel)

### Exécution du modèle de stock picking

Pour exécuter le modèle complet :

```bash
python run_stock_picker.py
```

Options disponibles :
```
  -c, --config CONFIG       Chemin vers le fichier de configuration
  -t, --tickers TICKERS     Liste de tickers séparés par des virgules
  -m, --market MARKET       Ticker de l'indice de marché (défaut: ^GSPC)
  --fundamental-weight W    Poids pour le score fondamental (0-1)
  --technical-weight W      Poids pour le score technique (0-1)
  --start-date START_DATE   Date de début (format YYYY-MM-DD)
  --end-date END_DATE       Date de fin (format YYYY-MM-DD)
  --skip-fetch              Sauter l'étape de collecte des données
  --skip-preprocessing      Sauter l'étape de prétraitement
```

### Optimisation de portefeuille

Pour optimiser un portefeuille basé sur les scores du modèle :

```bash
python optimize_portfolio.py
```

Options disponibles :
```
  -c, --config CONFIG       Chemin vers le fichier de configuration
  -t, --tickers TICKERS     Liste de tickers séparés par des virgules
  -m, --market MARKET       Ticker de l'indice de marché (défaut: ^GSPC)
  --method {markowitz,score-based,risk-parity}
                            Méthode d'optimisation du portefeuille
  --target-return TARGET    Rendement cible pour l'optimisation
  --risk-free-rate RATE     Taux sans risque annualisé (défaut: 0.025)
  --max-weight MAX          Poids maximum par actif (défaut: 0.15)
  --min-weight MIN          Poids minimum par actif (défaut: 0.01)
  --backtest                Effectuer un backtest du portefeuille optimisé
  --rebalance {D,W,M,Q,Y}   Fréquence de rééquilibrage (défaut: M)
  --plot                    Générer des graphiques pour visualiser les résultats
  --top-n N                 Limiter l'univers aux N meilleures actions
```

### Exemples d'utilisation

1. Scoring des actions du S&P 500 :

```bash
python run_stock_picker.py -t AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,JNJ,V
```

2. Optimisation d'un portefeuille avec la méthode de Markowitz :

```bash
python optimize_portfolio.py --method markowitz --top-n 20 --backtest --plot
```

3. Backtest d'un portefeuille optimisé avec rééquilibrage trimestriel :

```bash
python optimize_portfolio.py --method score-based --backtest --rebalance Q --plot
```

4. Utilisation de notebooks pour l'analyse exploratoire :

```bash
jupyter notebook notebooks/stock_analysis_example.ipynb
```

## Notebooks

Le projet inclut plusieurs notebooks Jupyter pour l'exploration et l'analyse :

- `notebooks/stock_analysis_example.ipynb` : Exemple d'analyse des actions avec le modèle
- `notebooks/portfolio_optimization_example.ipynb` : Exploration des méthodes d'optimisation de portefeuille

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.
