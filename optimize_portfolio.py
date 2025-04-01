#!/usr/bin/env python
"""
Script pour l'optimisation de portefeuille basée sur les résultats du modèle de stock picking.
Permet de construire des portefeuilles optimisés selon différentes méthodes.
"""
import os
import sys
import argparse
import logging
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Optimise un portefeuille d'actions basé sur le modèle de stock picking")
    
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
        choices=["markowitz", "score-based", "risk-parity"],
        default="score-based",
        help="Méthode d'optimisation du portefeuille"
    )
    
    parser.add_argument(
        "--target-return",
        type=float,
        help="Rendement cible pour l'optimisation (méthode Markowitz uniquement)"
    )
    
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.025,
        help="Taux sans risque annualisé pour le calcul du ratio de Sharpe"
    )
    
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.15,
        help="Poids maximum par actif (0-1)"
    )
    
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.01,
        help="Poids minimum par actif (0-1)"
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
        "--backtest",
        action="store_true",
        help="Effectuer un backtest du portefeuille optimisé"
    )
    
    parser.add_argument(
        "--rebalance",
        type=str,
        choices=["D", "W", "M", "Q", "Y"],
        default="M",
        help="Fréquence de rééquilibrage pour le backtest (D=quotidien, W=hebdomadaire, M=mensuel, etc.)"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Générer des graphiques pour visualiser les résultats"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Limiter l'univers d'optimisation aux N meilleures actions selon le score multifactoriel"
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
    default_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
        'JPM', 'BAC', 'JNJ', 'PG', 'V', 'MA', 'UNH', 'HD'
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
        from src.data.collectors import AlphaVantageCollector
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

def calculate_scores(processed_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Calcule les scores pour les actions.
    
    Args:
        processed_data: Dictionnaire contenant les données prétraitées
        
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
        factor_weights={'fundamental': 0.7, 'technical': 0.3}
    )
    
    # Calculer les scores fondamentaux
    logger.info("Calcul des scores fondamentaux...")
    fundamental_scores = fundamental_scorer.score_stocks(processed_data['fundamental'])
    
    # Calculer les scores techniques
    logger.info("Calcul des scores techniques...")
    technical_scores = technical_scorer.score_stocks(
        processed_data['price'], 
        processed_data['market']
    )
    
    # Calculer les scores multifactoriels
    logger.info("Calcul des scores multifactoriels...")
    multifactor_scores = multifactor_scorer.score_stocks(
        processed_data['fundamental'],
        processed_data['price'],
        None,  # Pas de données qualitatives pour l'instant
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
    method: str = "score-based",
    risk_free_rate: float = 0.025,
    target_return: Optional[float] = None,
    max_weight: float = 0.15,
    min_weight: float = 0.01,
    top_n: Optional[int] = None
) -> Dict[str, Any]:
    """
    Optimise un portefeuille basé sur les données prétraitées et les scores.
    
    Args:
        processed_data: Dictionnaire contenant les données prétraitées
        scores: Dictionnaire contenant les scores calculés
        method: Méthode d'optimisation ('markowitz', 'score-based', 'risk-parity')
        risk_free_rate: Taux sans risque annualisé
        target_return: Rendement cible annualisé (pour la méthode Markowitz)
        max_weight: Poids maximum par actif
        min_weight: Poids minimum par actif
        top_n: Nombre maximum d'actions à considérer (prend les meilleures selon le score multifactoriel)
        
    Returns:
        Dictionnaire contenant les résultats de l'optimisation
    """
    logger.info(f"Optimisation du portefeuille avec la méthode '{method}'...")
    
    # Initialiser l'optimiseur
    optimizer = PortfolioOptimizer(output_dir="data/results")
    
    # Filtrer les données si top_n est spécifié
    price_data = processed_data['price']
    multifactor_scores = scores['multifactor']
    
    if top_n is not None and top_n > 0:
        # Sélectionner les N meilleures actions selon le score multifactoriel
        top_tickers = multifactor_scores.sort_values('overall_score', ascending=False).head(top_n)['ticker'].tolist()
        logger.info(f"Limitation de l'univers d'optimisation aux {top_n} meilleures actions: {top_tickers}")
        
        # Filtrer les données de prix
        price_data = {ticker: df for ticker, df in price_data.items() if ticker in top_tickers}
        
        # Filtrer les scores
        multifactor_scores = multifactor_scores[multifactor_scores['ticker'].isin(top_tickers)]
    
    # Optimiser le portefeuille
    if method == "markowitz":
        optimal_portfolio = optimizer.optimize(
            price_data,
            multifactor_scores,
            method="markowitz",
            risk_free_rate=risk_free_rate,
            target_return=target_return,
            max_weight=max_weight,
            min_weight=min_weight
        )
    elif method == "score-based":
        optimal_portfolio = optimizer.optimize(
            price_data,
            multifactor_scores,
            method="score_based",
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            min_weight=min_weight
        )
    elif method == "risk-parity":
        optimal_portfolio = optimizer.optimize(
            price_data,
            multifactor_scores,
            method="risk_parity",
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            min_weight=min_weight
        )
    else:
        logger.error(f"Méthode d'optimisation inconnue: {method}")
        return {}
    
    # Extraire les poids optimaux
    weights = dict(zip(
        optimal_portfolio['ticker'],
        optimal_portfolio['weight']
    ))
    
    # Ne garder que les poids significatifs (> 1%)
    significant_weights = {ticker: weight for ticker, weight in weights.items() if weight > min_weight}
    
    # Normaliser pour s'assurer que la somme est 1
    total_weight = sum(significant_weights.values())
    normalized_weights = {ticker: weight / total_weight for ticker, weight in significant_weights.items()}
    
    # Extraire les métriques de performance
    performance_metrics = optimal_portfolio[
        ['expected_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'effective_assets']
    ].iloc[0].to_dict()
    
    # Organiser les résultats
    results = {
        'portfolio': optimal_portfolio,
        'weights': normalized_weights,
        'metrics': performance_metrics
    }
    
    logger.info(f"Optimisation terminée, {len(normalized_weights)} actions dans le portefeuille")
    return results

def backtest_portfolio(
    processed_data: Dict[str, Dict[str, pd.DataFrame]],
    weights: Dict[str, float],
    start_date: Optional[str] = None,
    rebalance_frequency: str = "M"
) -> pd.DataFrame:
    """
    Effectue un backtest du portefeuille optimisé.
    
    Args:
        processed_data: Dictionnaire contenant les données prétraitées
        weights: Dictionnaire des poids optimaux par ticker
        start_date: Date de début du backtest (format YYYY-MM-DD)
        rebalance_frequency: Fréquence de rééquilibrage ('D', 'W', 'M', 'Q', 'Y')
        
    Returns:
        DataFrame contenant les résultats du backtest
    """
    logger.info(f"Backtest du portefeuille optimisé depuis {start_date}...")
    
    # Initialiser l'optimiseur
    optimizer = PortfolioOptimizer(output_dir="data/results")
    
    # Effectuer le backtest
    backtest_results = optimizer.backtest_portfolio(
        processed_data['price'],
        weights,
        start_date=start_date,
        rebalance_frequency=rebalance_frequency
    )
    
    # Extraire les métriques de performance
    backtest_metrics = {
        'Total Return': backtest_results['total_return'].iloc[0],
        'Annualized Return': backtest_results['annualized_return'].iloc[0],
        'Annualized Volatility': backtest_results['annualized_volatility'].iloc[0],
        'Sharpe Ratio': backtest_results['sharpe_ratio'].iloc[0],
        'Maximum Drawdown': backtest_results['max_drawdown'].iloc[0]
    }
    
    # Afficher les métriques
    logger.info("Résultats du backtest:")
    for metric, value in backtest_metrics.items():
        if metric in ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown']:
            logger.info(f"  {metric}: {value*100:.2f}%")
        else:
            logger.info(f"  {metric}: {value:.2f}")
    
    return backtest_results

def plot_portfolio(portfolio: pd.DataFrame, weights: Dict[str, float], title: str) -> None:
    """
    Génère un graphique en camembert pour visualiser la composition du portefeuille.
    
    Args:
        portfolio: DataFrame contenant les données du portefeuille
        weights: Dictionnaire des poids par ticker
        title: Titre du graphique
    """
    # Trier les poids par ordre décroissant
    sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
    
    # Créer une figure
    plt.figure(figsize=(12, 8))
    
    # Visualiser les poids sous forme de camembert
    plt.pie(
        sorted_weights.values(), 
        labels=sorted_weights.keys(), 
        autopct='%1.1f%%', 
        startangle=90, 
        shadow=True
    )
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(title, fontsize=16)
    
    # Sauvegarder le graphique
    output_dir = "data/results/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{output_dir}/portfolio_allocation_{date_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Graphique de composition du portefeuille sauvegardé dans {filename}")
    
    plt.close()

def plot_backtest_results(backtest_results: pd.DataFrame, market_data: pd.DataFrame, title: str) -> None:
    """
    Génère un graphique pour visualiser les résultats du backtest.
    
    Args:
        backtest_results: DataFrame contenant les résultats du backtest
        market_data: DataFrame contenant les données de l'indice de marché
        title: Titre du graphique
    """
    # Aligner les dates
    common_dates = backtest_results.index.intersection(market_data.index)
    
    # Normaliser les performances
    portfolio_perf = backtest_results.loc[common_dates, 'portfolio_value'] / backtest_results.loc[common_dates[0], 'portfolio_value']
    market_perf = market_data.loc[common_dates, 'Close'] / market_data.loc[common_dates[0], 'Close']
    
    # Calculer la surperformance
    outperformance = portfolio_perf - market_perf
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Performance absolue
    ax1.plot(common_dates, portfolio_perf, label='Portefeuille', linewidth=2)
    ax1.plot(common_dates, market_perf, label='Marché (S&P 500)', linewidth=2, alpha=0.7)
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Performance (base 100)', fontsize=14)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Surperformance
    ax2.bar(common_dates, outperformance * 100, alpha=0.7, color='green', width=5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Surperformance (%)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.xlabel('Date', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    output_dir = "data/results/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{output_dir}/backtest_results_{date_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Graphique des résultats du backtest sauvegardé dans {filename}")
    
    plt.close()

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
    
    # Dates
    start_date = args.start_date
    end_date = args.end_date
    
    # Ticker de marché
    market_ticker = args.market
    
    # Alpha Vantage
    use_alpha_vantage = False
    alpha_vantage_key = None
    
    if 'ALPHA_VANTAGE' in config and 'api_key' in config['ALPHA_VANTAGE']:
        use_alpha_vantage = True
        alpha_vantage_key = config['ALPHA_VANTAGE']['api_key']
    
    # Collecte et prétraitement des données
    data = {}
    processed_data = {}
    
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
    
    if not args.skip_preprocessing:
        processed_data = preprocess_data(data)
    else:
        logger.info("Étape de prétraitement des données ignorée")
        # TODO: Charger les données prétraitées existantes
    
    # Calculer les scores
    scores = calculate_scores(processed_data)
    
    # Afficher un résumé des résultats du scoring
    logger.info("Top 10 actions selon le score multifactoriel:")
    top_stocks = scores['multifactor'].sort_values('overall_score', ascending=False).head(10)
    for i, (_, row) in enumerate(top_stocks.iterrows()):
        logger.info(f"{i+1}. {row['ticker']} - Score: {row['overall_score']:.3f}")
    
    # Optimiser le portefeuille
    optimization_results = optimize_portfolio(
        processed_data,
        scores,
        method=args.method.replace('-', '_'),
        risk_free_rate=args.risk_free_rate,
        target_return=args.target_return,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        top_n=args.top_n
    )
    
    # Afficher les résultats de l'optimisation
    logger.info(f"\nPortefeuille optimal selon la méthode '{args.method}':")
    for ticker, weight in sorted(optimization_results['weights'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {ticker}: {weight*100:.2f}%")
    
    logger.info("\nMétriques de performance:")
    for metric, value in optimization_results['metrics'].items():
        if metric in ['expected_return', 'volatility', 'max_drawdown']:
            logger.info(f"  {metric.replace('_', ' ').title()}: {value*100:.2f}%")
        else:
            logger.info(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
    
    # Effectuer un backtest si demandé
    backtest_results = None
    if args.backtest:
        # Déterminer la date de début du backtest (2 ans par défaut)
        backtest_start = args.start_date
        if not backtest_start:
            backtest_start = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
        
        backtest_results = backtest_portfolio(
            processed_data,
            optimization_results['weights'],
            start_date=backtest_start,
            rebalance_frequency=args.rebalance
        )
    
    # Générer des graphiques si demandé
    if args.plot:
        # Graphique de la composition du portefeuille
        plot_portfolio(
            optimization_results['portfolio'],
            optimization_results['weights'],
            f"Composition du portefeuille optimisé ({args.method})"
        )
        
        # Graphique des résultats du backtest
        if backtest_results is not None and market_ticker in processed_data['price']:
            plot_backtest_results(
                backtest_results,
                processed_data['price'][market_ticker],
                f"Performance du portefeuille vs Marché (Rééquilibrage: {args.rebalance})"
            )
    
    logger.info("Exécution terminée avec succès")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Erreur lors de l'exécution: {str(e)}")
        sys.exit(1)
