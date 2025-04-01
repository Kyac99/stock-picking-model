#!/usr/bin/env python
"""
Script principal pour exécuter l'optimisation de portefeuille basée sur les résultats du modèle de stock picking.
Utilise les scores générés par le modèle et optimise un portefeuille selon différentes méthodes.
"""
import os
import sys
import logging
import argparse
import configparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

# Ajouter le répertoire src au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.collectors import YahooFinanceCollector
from src.data.preprocessors import PriceDataPreprocessor, FundamentalDataPreprocessor
from src.models.scoring import FundamentalScorer, TechnicalScorer, MultifactorScorer
from src.models.portfolio_optimization import PortfolioOptimizer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/portfolio_optimizer_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Exécute l'optimisation de portefeuille")
    
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
        "--method",
        type=str,
        choices=["markowitz", "score_based", "risk_parity", "all"],
        default="all",
        help="Méthode d'optimisation de portefeuille à utiliser"
    )
    
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.025,  # 2.5% par défaut
        help="Taux sans risque annualisé pour les calculs (0-1)"
    )
    
    parser.add_argument(
        "--target-return",
        type=float,
        help="Rendement cible annualisé pour l'optimisation de Markowitz (si non spécifié, maximise le ratio de Sharpe)"
    )
    
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.15,  # 15% par défaut
        help="Poids maximum par action dans le portefeuille (0-1)"
    )
    
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.01,  # 1% par défaut
        help="Poids minimum par action dans le portefeuille (0-1)"
    )
    
    parser.add_argument(
        "--top-stocks",
        type=int,
        default=20,
        help="Nombre d'actions à considérer pour l'optimisation (prises selon le score multifactoriel)"
    )
    
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Effectuer un backtest du portefeuille optimisé"
    )
    
    parser.add_argument(
        "--backtest-period",
        type=int,
        default=2,
        help="Nombre d'années pour le backtest"
    )
    
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        choices=["D", "W", "M", "Q", "Y"],
        default="M",
        help="Fréquence de rééquilibrage pour le backtest (D=daily, W=weekly, M=monthly, Q=quarterly, Y=yearly)"
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
    
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Sauter l'étape de scoring (utiliser les scores existants)"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Générer des graphiques pour visualiser les résultats"
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
    default_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'INTC', 'DIS', 'ADBE', 'NFLX'
    ]
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
        from src.data.collectors import AlphaVantageCollector
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
    processed_data: Dict[str, Dict[str, pd.DataFrame]],
    fundamental_weight: float = 0.7,
    technical_weight: float = 0.3
) -> Dict[str, pd.DataFrame]:
    """
    Calcule les scores pour les actions.
    
    Args:
        processed_data: Dictionnaire contenant les données prétraitées
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
        factor_weights={"fundamental": fundamental_weight, "technical": technical_weight},
        fundamental_scorer=fundamental_scorer,
        technical_scorer=technical_scorer
    )
    
    # Calculer les scores fondamentaux
    logger.info("Calcul des scores fondamentaux...")
    fundamental_scores = fundamental_scorer.score_stocks(processed_data['fundamental'])
    
    # Calculer les scores techniques
    logger.info("Calcul des scores techniques...")
    technical_scores = technical_scorer.score_stocks(processed_data['price'], processed_data['market'])
    
    # Calculer les scores multifactoriels
    logger.info("Calcul des scores multifactoriels...")
    multifactor_scores = multifactor_scorer.score_stocks(
        processed_data['fundamental'],
        processed_data['price'],
        None,  # Pas de données qualitatives pour le moment
        processed_data['market']
    )
    
    # Organiser les scores
    scores = {
        'fundamental': fundamental_scores,
        'technical': technical_scores,
        'multifactor': multifactor_scores
    }
    
    logger.info("Calcul des scores terminé")
    return scores

def optimize_portfolio(
    processed_data: Dict[str, Dict[str, pd.DataFrame]],
    scores: Dict[str, pd.DataFrame],
    method: str,
    top_n: int = 20,
    risk_free_rate: float = 0.025,
    target_return: Optional[float] = None,
    max_weight: float = 0.15,
    min_weight: float = 0.01
) -> pd.DataFrame:
    """
    Optimise un portefeuille selon la méthode spécifiée.
    
    Args:
        processed_data: Dictionnaire contenant les données prétraitées
        scores: Dictionnaire contenant les scores calculés
        method: Méthode d'optimisation ('markowitz', 'score_based', 'risk_parity')
        top_n: Nombre d'actions à considérer (prises selon le score multifactoriel)
        risk_free_rate: Taux sans risque annualisé
        target_return: Rendement cible annualisé (si None, maximise le ratio de Sharpe)
        max_weight: Poids maximum par action
        min_weight: Poids minimum par action
        
    Returns:
        DataFrame contenant le portefeuille optimisé
    """
    logger.info(f"Optimisation du portefeuille selon la méthode '{method}'...")
    
    # Initialiser l'optimiseur
    optimizer = PortfolioOptimizer(output_dir="data/results")
    
    # Sélectionner les meilleures actions selon le score multifactoriel
    if top_n > 0 and top_n < len(scores['multifactor']):
        top_stocks = scores['multifactor'].sort_values('overall_score', ascending=False).head(top_n)['ticker'].tolist()
        logger.info(f"Sélection des {top_n} meilleures actions selon le score multifactoriel: {top_stocks}")
    else:
        top_stocks = scores['multifactor']['ticker'].tolist()
        logger.info(f"Utilisation de toutes les actions ({len(top_stocks)}) pour l'optimisation")
    
    # Filtrer les données de prix pour ces actions
    price_data = {ticker: processed_data['price'][ticker] for ticker in top_stocks if ticker in processed_data['price']}
    
    # Optimiser le portefeuille selon la méthode spécifiée
    portfolio = optimizer.optimize(
        price_data,
        scores['multifactor'][scores['multifactor']['ticker'].isin(top_stocks)],
        method=method,
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        max_weight=max_weight,
        min_weight=min_weight
    )
    
    # Afficher les résultats (actions avec un poids > 1%)
    significant_weights = portfolio[portfolio['weight'] > 0.01].sort_values('weight', ascending=False)
    logger.info(f"Portefeuille optimisé ({method}) - {len(significant_weights)} actions avec poids > 1%:")
    
    for _, row in significant_weights.iterrows():
        logger.info(f"  {row['ticker']}: {row['weight']*100:.2f}%")
    
    # Extraire et afficher les métriques de performance
    metrics_cols = ['expected_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'effective_assets']
    if all(metric in portfolio.columns for metric in metrics_cols):
        metrics = portfolio[metrics_cols].iloc[0].to_dict()
        
        logger.info("\nMétriques de performance:")
        for metric, value in metrics.items():
            if metric in ['expected_return', 'volatility', 'max_drawdown']:
                logger.info(f"  {metric.replace('_', ' ').title()}: {value*100:.2f}%")
            else:
                logger.info(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
    
    return portfolio

def backtest_portfolio(
    portfolio: pd.DataFrame,
    processed_data: Dict[str, Dict[str, pd.DataFrame]],
    start_date: Optional[str] = None,
    rebalance_frequency: str = 'M'
) -> pd.DataFrame:
    """
    Réalise un backtest du portefeuille optimisé.
    
    Args:
        portfolio: DataFrame contenant le portefeuille optimisé
        processed_data: Dictionnaire contenant les données prétraitées
        start_date: Date de début du backtest (format YYYY-MM-DD)
        rebalance_frequency: Fréquence de rééquilibrage ('D', 'W', 'M', 'Q', 'Y')
        
    Returns:
        DataFrame contenant les résultats du backtest
    """
    logger.info(f"Backtest du portefeuille optimisé depuis {start_date or 'le début des données'}...")
    
    # Extraire les poids optimaux
    weights = dict(zip(
        portfolio['ticker'], 
        portfolio['weight']
    ))
    
    # Ne garder que les poids significatifs (> 1%)
    weights = {ticker: weight for ticker, weight in weights.items() if weight > 0.01}
    
    # Normaliser pour s'assurer que la somme est 1
    total_weight = sum(weights.values())
    weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
    
    # Initialiser l'optimiseur (pour utiliser la méthode de backtest)
    optimizer = PortfolioOptimizer(output_dir="data/results")
    
    # Réaliser le backtest
    backtest_results = optimizer.backtest_portfolio(
        processed_data['price'],
        weights,
        start_date=start_date,
        rebalance_frequency=rebalance_frequency
    )
    
    # Extraire et afficher les métriques de performance du backtest
    metrics_cols = ['total_return', 'annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown']
    if all(metric in backtest_results.columns for metric in metrics_cols):
        metrics = {
            'Total Return': backtest_results['total_return'].iloc[0],
            'Annualized Return': backtest_results['annualized_return'].iloc[0],
            'Annualized Volatility': backtest_results['annualized_volatility'].iloc[0],
            'Sharpe Ratio': backtest_results['sharpe_ratio'].iloc[0],
            'Maximum Drawdown': backtest_results['max_drawdown'].iloc[0]
        }
        
        logger.info("\nRésultats du backtest:")
        for metric, value in metrics.items():
            if metric in ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown']:
                logger.info(f"  {metric}: {value*100:.2f}%")
            else:
                logger.info(f"  {metric}: {value:.2f}")
    
    return backtest_results

def plot_results(
    portfolio: pd.DataFrame,
    backtest_results: Optional[pd.DataFrame] = None,
    market_data: Optional[pd.DataFrame] = None,
    method: str = 'markowitz'
):
    """
    Génère des visualisations pour les résultats de l'optimisation et du backtest.
    
    Args:
        portfolio: DataFrame contenant le portefeuille optimisé
        backtest_results: DataFrame contenant les résultats du backtest
        market_data: DataFrame contenant les données de l'indice de marché
        method: Méthode d'optimisation utilisée
    """
    # Importer les bibliothèques de visualisation
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Créer le répertoire de graphiques s'il n'existe pas
    os.makedirs("data/plots", exist_ok=True)
    
    # Configurer le style des graphiques
    plt.style.use('seaborn-darkgrid')
    sns.set(font_scale=1.2)
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # 1. Visualisation de l'allocation du portefeuille
    if 'weight' in portfolio.columns:
        # Filtrer les poids significatifs
        significant = portfolio[portfolio['weight'] > 0.01].sort_values('weight', ascending=False)
        
        plt.figure(figsize=(14, 10))
        
        # Graphique à barres pour les poids
        sns.barplot(x='ticker', y='weight', data=significant, palette='viridis')
        
        plt.title(f'Allocation du portefeuille optimisé ({method.capitalize()})', fontsize=16)
        plt.xlabel('Action', fontsize=14)
        plt.ylabel('Poids dans le portefeuille', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plt.savefig(f"data/plots/portfolio_allocation_{method}_{datetime.now().strftime('%Y%m%d')}.png")
        logger.info(f"Graphique d'allocation sauvegardé dans data/plots/portfolio_allocation_{method}_{datetime.now().strftime('%Y%m%d')}.png")
    
    # 2. Visualisation du backtest si disponible
    if backtest_results is not None and 'portfolio_value' in backtest_results.columns:
        # Normaliser la performance
        performance = backtest_results['portfolio_value'] / backtest_results['portfolio_value'].iloc[0]
        
        plt.figure(figsize=(14, 8))
        
        # Tracer la performance du portefeuille
        plt.plot(performance.index, performance, label='Portefeuille optimisé', linewidth=2)
        
        # Ajouter l'indice de marché si disponible
        if market_data is not None and 'Close' in market_data.columns:
            # S'assurer que les dates correspondent
            market_subset = market_data.loc[market_data.index >= performance.index[0], 'Close']
            
            if not market_subset.empty:
                # Normaliser la performance du marché
                market_performance = market_subset / market_subset.iloc[0]
                
                # Ajouter au graphique
                plt.plot(market_performance.index, market_performance, label='Indice de référence', linestyle='--', color='gray')
        
        plt.title(f'Performance du portefeuille optimisé ({method.capitalize()})', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Performance relative (base 1)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plt.savefig(f"data/plots/portfolio_performance_{method}_{datetime.now().strftime('%Y%m%d')}.png")
        logger.info(f"Graphique de performance sauvegardé dans data/plots/portfolio_performance_{method}_{datetime.now().strftime('%Y%m%d')}.png")

def main():
    """Fonction principale."""
    # Créer les répertoires nécessaires
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Parser les arguments de ligne de commande
    args = parse_args()
    
    # Charger la configuration
    config = load_config(args.config)
    
    # Obtenir la liste des tickers
    tickers = get_tickers(args, config)
    
    # Récupérer les paramètres de la configuration si non spécifiés en ligne de commande
    if 'TIME_PERIODS' in config:
        if args.start_date is None and 'default_start_date' in config['TIME_PERIODS']:
            args.start_date = config['TIME_PERIODS']['default_start_date']
        if args.end_date is None and 'default_end_date' in config['TIME_PERIODS']:
            args.end_date = config['TIME_PERIODS']['default_end_date']
    
    if 'PORTFOLIO_OPTIMIZATION' in config:
        if args.risk_free_rate is None and 'risk_free_rate' in config['PORTFOLIO_OPTIMIZATION']:
            args.risk_free_rate = config['PORTFOLIO_OPTIMIZATION'].getfloat('risk_free_rate')
        if args.target_return is None and 'target_return' in config['PORTFOLIO_OPTIMIZATION']:
            target_return = config['PORTFOLIO_OPTIMIZATION']['target_return']
            if target_return.lower() != 'none':
                args.target_return = float(target_return)
        if args.max_weight is None and 'max_weight_per_asset' in config['PORTFOLIO_OPTIMIZATION']:
            args.max_weight = config['PORTFOLIO_OPTIMIZATION'].getfloat('max_weight_per_asset')
        if args.min_weight is None and 'min_weight_per_asset' in config['PORTFOLIO_OPTIMIZATION']:
            args.min_weight = config['PORTFOLIO_OPTIMIZATION'].getfloat('min_weight_per_asset')
    
    # Alpha Vantage
    use_alpha_vantage = False
    alpha_vantage_key = None
    
    if 'DATA_SOURCES' in config and 'use_alpha_vantage' in config['DATA_SOURCES']:
        use_alpha_vantage = config['DATA_SOURCES'].getboolean('use_alpha_vantage')
    
    if 'ALPHA_VANTAGE' in config and 'api_key' in config['ALPHA_VANTAGE']:
        alpha_vantage_key = config['ALPHA_VANTAGE']['api_key']
        if alpha_vantage_key == 'YOUR_API_KEY_HERE':
            alpha_vantage_key = None
            use_alpha_vantage = False
    
    # Collecter les données
    data = {}
    if not args.skip_fetch:
        data = collect_data(
            tickers,
            args.market,
            args.start_date,
            args.end_date,
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
    scores = {}
    if not args.skip_scoring:
        # Récupérer les poids des facteurs depuis la configuration
        fundamental_weight = 0.7
        technical_weight = 0.3
        
        if 'SCORING' in config:
            if 'fundamental_weight' in config['SCORING']:
                fundamental_weight = config['SCORING'].getfloat('fundamental_weight')
            if 'technical_weight' in config['SCORING']:
                technical_weight = config['SCORING'].getfloat('technical_weight')
        
        scores = calculate_scores(
            processed_data,
            fundamental_weight,
            technical_weight
        )
    else:
        logger.info("Étape de scoring ignorée")
        # TODO: Charger les scores existants
    
    # Optimiser le portefeuille
    portfolios = {}
    
    if args.method == 'all':
        # Optimiser avec toutes les méthodes
        methods = ['markowitz', 'score_based', 'risk_parity']
        for method in methods:
            portfolios[method] = optimize_portfolio(
                processed_data,
                scores,
                method,
                top_n=args.top_stocks,
                risk_free_rate=args.risk_free_rate,
                target_return=args.target_return,
                max_weight=args.max_weight,
                min_weight=args.min_weight
            )
    else:
        # Optimiser avec la méthode spécifiée
        portfolios[args.method] = optimize_portfolio(
            processed_data,
            scores,
            args.method,
            top_n=args.top_stocks,
            risk_free_rate=args.risk_free_rate,
            target_return=args.target_return,
            max_weight=args.max_weight,
            min_weight=args.min_weight
        )
    
    # Backtest si demandé
    if args.backtest:
        backtest_results = {}
        
        # Calculer la date de début du backtest
        backtest_start = None
        if args.backtest_period > 0:
            backtest_start = (datetime.now() - timedelta(days=365 * args.backtest_period)).strftime('%Y-%m-%d')
        
        for method, portfolio in portfolios.items():
            backtest_results[method] = backtest_portfolio(
                portfolio,
                processed_data,
                start_date=backtest_start,
                rebalance_frequency=args.rebalance_frequency
            )
    
    # Visualisation si demandée
    if args.plot:
        for method, portfolio in portfolios.items():
            backtest_result = backtest_results.get(method) if args.backtest else None
            plot_results(
                portfolio,
                backtest_result,
                processed_data.get('market'),
                method
            )
    
    logger.info("Exécution terminée avec succès")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Erreur lors de l'exécution: {str(e)}")
        sys.exit(1)
