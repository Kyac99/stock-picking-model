#!/usr/bin/env python
"""
Script principal pour exécuter le modèle de stock picking.
Collecte les données, effectue l'analyse et génère les scores.
"""
import os
import sys
import logging
import argparse
import configparser
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any

# Ajouter le répertoire src au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Créer les répertoires nécessaires
os.makedirs("logs", exist_ok=True)

from src.data.collectors import YahooFinanceCollector, AlphaVantageCollector
from src.data.preprocessors import PriceDataPreprocessor, FundamentalDataPreprocessor
from src.models.scoring import FundamentalScorer, TechnicalScorer, MultifactorScorer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/stock_picker_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Exécute le modèle de stock picking")
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.ini",
        help="Chemin vers le fichier de configuration"
    )
    
    parser.add_argument(
        "-t", "--tickers",
        type=str,
        help="Liste de tickers séparés par des virgules (override la liste dans le fichier de config)"
    )
    
    parser.add_argument(
        "-m", "--market",
        type=str,
        default="^GSPC",  # S&P 500 par défaut
        help="Ticker de l'indice de marché à utiliser pour la force relative"
    )
    
    parser.add_argument(
        "--fundamental-weight",
        type=float,
        default=0.7,
        help="Poids pour le score fondamental (0-1)"
    )
    
    parser.add_argument(
        "--technical-weight",
        type=float,
        default=0.3,
        help="Poids pour le score technique (0-1)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Date de début pour les données (format YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="Date de fin pour les données (format YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Sauter l'étape de collecte des données (utiliser les données existantes)"
    )
    
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Sauter l'étape de prétraitement des données (utiliser les données prétraitées existantes)"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> configparser.ConfigParser:
    """Charge la configuration depuis un fichier INI."""
    logger.info(f"Chargement de la configuration depuis {config_path}")
    
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_path):
        logger.warning(f"Fichier de configuration {config_path} non trouvé, utilisation des valeurs par défaut")
        return config
    
    config.read(config_path)
    return config

def get_tickers(args, config) -> List[str]:
    """Obtient la liste des tickers à analyser."""
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
        logger.info(f"Utilisation des tickers spécifiés en ligne de commande: {tickers}")
        return tickers
    
    if 'TICKERS' in config and 'symbols' in config['TICKERS']:
        tickers = [ticker.strip() for ticker in config['TICKERS']['symbols'].split(',')]
        logger.info(f"Utilisation des tickers du fichier de configuration: {tickers}")
        return tickers
    
    # Liste par défaut
    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V']
    logger.info(f"Aucun ticker spécifié, utilisation de la liste par défaut: {default_tickers}")
    return default_tickers

def collect_data(
    tickers: List[str],
    market_ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_alpha_vantage: bool = False,
    alpha_vantage_key: Optional[str] = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Collecte les données historiques et fondamentales pour une liste de tickers.
    
    Args:
        tickers: Liste des tickers à collecter
        market_ticker: Ticker de l'indice de marché
        start_date: Date de début (format YYYY-MM-DD)
        end_date: Date de fin (format YYYY-MM-DD)
        use_alpha_vantage: Si True, utilise Alpha Vantage en plus de Yahoo Finance
        alpha_vantage_key: Clé API Alpha Vantage (nécessaire si use_alpha_vantage=True)
        
    Returns:
        Dictionnaire contenant les données collectées par type et par ticker
    """
    logger.info(f"Collecte des données pour {len(tickers)} tickers")
    
    # Créer les répertoires de données s'ils n'existent pas
    os.makedirs("data/raw", exist_ok=True)
    
    # Initialiser le collecteur Yahoo Finance
    yf_collector = YahooFinanceCollector(output_dir="data/raw")
    
    # Collecter les données de prix
    logger.info("Collecte des données de prix historiques...")
    price_data = yf_collector.get_stock_data(tickers + [market_ticker], start_date, end_date)
    
    # Collecter les données fondamentales
    logger.info("Collecte des données fondamentales...")
    fundamental_data = {}
    for ticker in tickers:
        try:
            fundamental_data[ticker] = yf_collector.get_fundamentals([ticker])
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des données fondamentales pour {ticker}: {str(e)}")
    
    # Utiliser Alpha Vantage si spécifié
    if use_alpha_vantage and alpha_vantage_key:
        logger.info("Collecte de données supplémentaires depuis Alpha Vantage...")
        av_collector = AlphaVantageCollector(api_key=alpha_vantage_key, output_dir="data/raw")
        
        # Compléter avec des données plus détaillées
        for ticker in tickers:
            try:
                # Données fondamentales supplémentaires
                income_stmt = av_collector.get_fundamental_data(ticker, "INCOME_STATEMENT")
                balance_sheet = av_collector.get_fundamental_data(ticker, "BALANCE_SHEET")
                cash_flow = av_collector.get_fundamental_data(ticker, "CASH_FLOW")
                
                if ticker not in fundamental_data:
                    fundamental_data[ticker] = {}
                
                # Ajouter à la collection existante
                if not income_stmt.empty:
                    fundamental_data[ticker]['income_statement'] = income_stmt
                if not balance_sheet.empty:
                    fundamental_data[ticker]['balance_sheet'] = balance_sheet
                if not cash_flow.empty:
                    fundamental_data[ticker]['cash_flow'] = cash_flow
                    
            except Exception as e:
                logger.error(f"Erreur lors de la collecte des données Alpha Vantage pour {ticker}: {str(e)}")
    
    # Organiser les données
    data = {
        'price': price_data,
        'fundamental': fundamental_data,
        'market': price_data.get(market_ticker)
    }
    
    logger.info("Collecte des données terminée")
    return data

def preprocess_data(data: Dict[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Prétraite les données brutes.
    
    Args:
        data: Dictionnaire contenant les données brutes
        
    Returns:
        Dictionnaire contenant les données prétraitées
    """
    logger.info("Prétraitement des données...")
    
    # Créer les répertoires de données s'ils n'existent pas
    os.makedirs("data/processed", exist_ok=True)
    
    # Initialiser les préprocesseurs
    price_preprocessor = PriceDataPreprocessor(
        input_dir="data/raw",
        output_dir="data/processed"
    )
    
    fundamental_preprocessor = FundamentalDataPreprocessor(
        input_dir="data/raw",
        output_dir="data/processed"
    )
    
    # Prétraiter les données de prix
    processed_price = {}
    for ticker, price_df in data['price'].items():
        try:
            # Sauvegarder les données brutes si nécessaire
            filename = f"{ticker}_price.csv"
            price_df.to_csv(f"data/raw/{filename}")
            
            # Prétraiter
            processed_price[ticker] = price_preprocessor.preprocess(filename)
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement des données de prix pour {ticker}: {str(e)}")
    
    # Prétraiter les données fondamentales
    processed_fundamental = {}
    for ticker, fund_data in data['fundamental'].items():
        try:
            # Extraire et sauvegarder les données fondamentales
            income_stmt = fund_data.get('income_statement')
            balance_sheet = fund_data.get('balance_sheet')
            cash_flow = fund_data.get('cash_flow')
            
            # Sauvegarder au format CSV si nécessaire
            if isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
                income_stmt.to_csv(f"data/raw/{ticker}_income_stmt.csv")
            if isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty:
                balance_sheet.to_csv(f"data/raw/{ticker}_balance_sheet.csv")
            if isinstance(cash_flow, pd.DataFrame) and not cash_flow.empty:
                cash_flow.to_csv(f"data/raw/{ticker}_cash_flow.csv")
            
            # Prétraiter les données financières combinées
            if (isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty and
                isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty):
                
                ratios = fundamental_preprocessor.preprocess_financials(
                    f"{ticker}_income_stmt.csv",
                    f"{ticker}_balance_sheet.csv",
                    f"{ticker}_cash_flow.csv" if isinstance(cash_flow, pd.DataFrame) and not cash_flow.empty else None
                )
                
                processed_fundamental[ticker] = ratios
                
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement des données fondamentales pour {ticker}: {str(e)}")
    
    # Organiser les données prétraitées
    processed_data = {
        'price': processed_price,
        'fundamental': processed_fundamental,
        'market': data['market']  # Conserver les données de marché telles quelles
    }
    
    logger.info("Prétraitement des données terminé")
    return processed_data

def calculate_scores(
    data: Dict[str, Dict[str, pd.DataFrame]],
    fundamental_weight: float = 0.7,
    technical_weight: float = 0.3
) -> Dict[str, pd.DataFrame]:
    """
    Calcule les scores pour les actions.
    
    Args:
        data: Dictionnaire contenant les données prétraitées
        fundamental_weight: Poids pour le score fondamental (0-1)
        technical_weight: Poids pour le score technique (0-1)
        
    Returns:
        Dictionnaire contenant les différents scores calculés
    """
    logger.info("Calcul des scores...")
    
    # Créer le répertoire de résultats s'il n'existe pas
    os.makedirs("data/results", exist_ok=True)
    
    # Initialiser les scorers
    fundamental_scorer = FundamentalScorer(output_dir="data/results")
    technical_scorer = TechnicalScorer(output_dir="data/results")
    multifactor_scorer = MultifactorScorer(
        output_dir="data/results",
        fundamental_weight=fundamental_weight,
        technical_weight=technical_weight,
        fundamental_scorer=fundamental_scorer,
        technical_scorer=technical_scorer
    )
    
    # Calculer les scores fondamentaux
    logger.info("Calcul des scores fondamentaux...")
    fundamental_scores = fundamental_scorer.score_stocks(data['fundamental'])
    
    # Calculer les scores techniques
    logger.info("Calcul des scores techniques...")
    technical_scores = technical_scorer.score_stocks(data['price'], data['market'])
    
    # Calculer les scores multifactoriels
    logger.info("Calcul des scores multifactoriels...")
    multifactor_scores = multifactor_scorer.score_stocks(
        data['fundamental'],
        data['price'],
        data['market']
    )
    
    # Organiser les scores
    scores = {
        'fundamental': fundamental_scores,
        'technical': technical_scores,
        'multifactor': multifactor_scores
    }
    
    logger.info("Calcul des scores terminé")
    return scores

def main():
    """Fonction principale."""
    # Parser les arguments de ligne de commande
    args = parse_args()
    
    # Charger la configuration
    config = load_config(args.config)
    
    # Obtenir la liste des tickers
    tickers = get_tickers(args, config)
    
    # Dates
    start_date = args.start_date
    end_date = args.end_date
    
    # Poids pour les scores
    fundamental_weight = args.fundamental_weight
    technical_weight = args.technical_weight
    
    # Ticker de marché
    market_ticker = args.market
    
    # Alpha Vantage
    use_alpha_vantage = False
    alpha_vantage_key = None
    
    if 'ALPHA_VANTAGE' in config and 'api_key' in config['ALPHA_VANTAGE']:
        use_alpha_vantage = True
        alpha_vantage_key = config['ALPHA_VANTAGE']['api_key']
    
    # Collecter les données
    data = {}
    if not args.skip_fetch:
        data = collect_data(
            tickers,
            market_ticker,
            start_date,
            end_date,
            use_alpha_vantage,
            alpha_vantage_key
        )
    else:
        logger.info("Étape de collecte des données ignorée")
    
    # Prétraiter les données
    processed_data = {}
    if not args.skip_preprocessing:
        processed_data = preprocess_data(data)
    else:
        logger.info("Étape de prétraitement des données ignorée")
        # TODO: Charger les données prétraitées existantes
    
    # Calculer les scores
    scores = calculate_scores(
        processed_data,
        fundamental_weight,
        technical_weight
    )
    
    # Afficher un résumé des résultats
    for score_type, score_df in scores.items():
        if not score_df.empty:
            logger.info(f"Top 5 actions selon le score {score_type}:")
            for i, (_, row) in enumerate(score_df.sort_values('overall_score', ascending=False).head(5).iterrows()):
                logger.info(f"{i+1}. {row['ticker']} - Score: {row['overall_score']:.3f}")
    
    logger.info("Exécution terminée avec succès")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Erreur lors de l'exécution: {str(e)}")
        sys.exit(1)
