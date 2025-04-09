#!/usr/bin/env python
"""
Script pour optimiser un portefeuille basé sur les résultats du modèle de stock picking.
Charge les scores générés par le modèle et optimise l'allocation de portefeuille.
"""
import os
import sys
import logging
import argparse
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any, Tuple

# Créer les répertoires nécessaires
os.makedirs("logs", exist_ok=True)

# Ajouter le répertoire src au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.collectors import YahooFinanceCollector
from src.data.preprocessors import PriceDataPreprocessor
from src.models.portfolio_optimization import PortfolioOptimizer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/portfolio_optimization_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Optimise un portefeuille basé sur les résultats du modèle de stock picking")
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.ini",
        help="Chemin vers le fichier de configuration"
    )
    
    parser.add_argument(
        "-s", "--scores",
        type=str,
        help="Chemin vers le fichier CSV des scores (si non spécifié, utilise le plus récent)"
    )
    
    parser.add_argument(
        "-m", "--method",
        type=str,
        choices=["markowitz", "score_based", "risk_parity"],
        default="score_based",
        help="Méthode d'optimisation de portefeuille"
    )
    
    parser.add_argument(
        "-n", "--num-stocks",
        type=int,
        default=20,
        help="Nombre d'actions à inclure dans l'univers d'optimisation"
    )
    
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.15,
        help="Poids maximum par action (0-1)"
    )
    
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.01,
        help="Poids minimum par action (0-1)"
    )
    
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.025,
        help="Taux sans risque annualisé (0-1)"
    )
    
    parser.add_argument(
        "--lookback",
        type=int,
        default=3,
        help="Période historique en années pour l'analyse"
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
        help="Période de backtest en années"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Générer des graphiques"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Répertoire de sortie pour les résultats"
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

def load_scores_file(scores_path: Optional[str] = None) -> pd.DataFrame:
    """
    Charge les scores du modèle depuis un fichier CSV.
    
    Args:
        scores_path: Chemin du fichier. Si None, utilise le plus récent.
        
    Returns:
        DataFrame des scores
    """
    if scores_path and os.path.exists(scores_path):
        logger.info(f"Chargement des scores depuis {scores_path}")
        return pd.read_csv(scores_path)
    
    # Chercher le fichier de scores le plus récent
    results_dir = "data/results"
    if not os.path.exists(results_dir):
        logger.error(f"Répertoire {results_dir} non trouvé")
        sys.exit(1)
    
    score_files = []
    for filename in os.listdir(results_dir):
        if filename.startswith("multifactor_scores_") and filename.endswith(".csv"):
            score_files.append(os.path.join(results_dir, filename))
    
    if not score_files:
        logger.error("Aucun fichier de scores trouvé")
        sys.exit(1)
    
    # Trier par date de modification (le plus récent en premier)
    most_recent_file = max(score_files, key=os.path.getmtime)
    logger.info(f"Utilisation du fichier de scores le plus récent: {most_recent_file}")
    
    return pd.read_csv(most_recent_file)

def collect_price_data(tickers: List[str], lookback_years: int) -> Dict[str, pd.DataFrame]:
    """
    Collecte les données de prix historiques pour les tickers spécifiés.
    
    Args:
        tickers: Liste des tickers à collecter
        lookback_years: Nombre d'années d'historique
        
    Returns:
        Dictionnaire des DataFrames de prix par ticker
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * lookback_years)
    
    logger.info(f"Collecte des données de prix pour {len(tickers)} actions (période: {start_date.date()} à {end_date.date()})...")
    
    # Créer les répertoires de données si nécessaires
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Initialiser le collecteur
    collector = YahooFinanceCollector(output_dir="data/raw")
    
    # Collecter les données
    price_data = collector.get_stock_data(
        tickers + ['^GSPC'],  # Inclure le S&P 500 comme référence
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Prétraiter les données
    logger.info("Prétraitement des données de prix...")
    preprocessor = PriceDataPreprocessor(
        input_dir="data/raw",
        output_dir="data/processed"
    )
    
    processed_data = {}
    for ticker in tickers + ['^GSPC']:
        if ticker in price_data:
            # Trouver le fichier correspondant
            import glob
            files = glob.glob(f"data/raw/{ticker}_*.csv")
            if files:
                filename = os.path.basename(files[0])
                processed_data[ticker] = preprocessor.preprocess(filename)
    
    return processed_data

def plot_portfolio_allocation(portfolio_df: pd.DataFrame, method: str, output_dir: str):
    """
    Génère un graphique de l'allocation du portefeuille.
    
    Args:
        portfolio_df: DataFrame du portefeuille optimisé
        method: Méthode d'optimisation utilisée
        output_dir: Répertoire de sortie pour les graphiques
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrer pour ne garder que les poids significatifs
    allocation = portfolio_df[portfolio_df['weight'] > 0.01].sort_values('weight', ascending=True)
    
    # Créer le graphique
    plt.figure(figsize=(10, 8))
    bars = plt.barh(allocation['ticker'], allocation['weight'], color='skyblue')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                 ha='left', va='center')
    
    # Formater l'axe des y pour afficher les pourcentages
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    plt.title(f'Allocation du portefeuille optimisé ({method})', fontsize=14)
    plt.xlabel('Poids dans le portefeuille')
    plt.ylabel('Action')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    date_str = datetime.now().strftime('%Y%m%d')
    plt.savefig(f"{output_dir}/portfolio_allocation_{method}_{date_str}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Graphique d'allocation sauvegardé dans {output_dir}/portfolio_allocation_{method}_{date_str}.png")

def plot_backtest_results(backtest_results: pd.DataFrame, benchmark: pd.Series, output_dir: str):
    """
    Génère un graphique des résultats du backtest.
    
    Args:
        backtest_results: DataFrame des résultats du backtest
        benchmark: Série de prix de l'indice de référence
        output_dir: Répertoire de sortie pour les graphiques
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Normaliser les performances
    portfolio_perf = backtest_results['portfolio_value'] / backtest_results['portfolio_value'].iloc[0]
    benchmark_perf = benchmark / benchmark.iloc[0]
    
    # Aligner les dates
    common_dates = portfolio_perf.index.intersection(benchmark_perf.index)
    portfolio_perf = portfolio_perf.loc[common_dates]
    benchmark_perf = benchmark_perf.loc[common_dates]
    
    # Calculer la surperformance
    outperformance = portfolio_perf - benchmark_perf
    
    # Création du graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Performance absolue
    ax1.plot(portfolio_perf.index, portfolio_perf, label='Portefeuille optimisé', color='green', linewidth=2)
    ax1.plot(benchmark_perf.index, benchmark_perf, label='S&P 500', color='blue', linewidth=1.5, linestyle='--')
    ax1.set_title('Performance du portefeuille vs S&P 500', fontsize=14)
    ax1.set_ylabel('Performance normalisée')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Surperformance
    ax2.fill_between(outperformance.index, outperformance, 0, where=outperformance>=0, color='green', alpha=0.3, label='Surperformance')
    ax2.fill_between(outperformance.index, outperformance, 0, where=outperformance<0, color='red', alpha=0.3, label='Sous-performance')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Surperformance')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    date_str = datetime.now().strftime('%Y%m%d')
    plt.savefig(f"{output_dir}/backtest_results_{date_str}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Graphique de backtest sauvegardé dans {output_dir}/backtest_results_{date_str}.png")

def export_portfolio_to_csv(portfolio_df: pd.DataFrame, method: str, output_dir: str) -> str:
    """
    Exporte l'allocation du portefeuille dans un fichier CSV.
    
    Args:
        portfolio_df: DataFrame du portefeuille optimisé
        method: Méthode d'optimisation utilisée
        output_dir: Répertoire de sortie
        
    Returns:
        Chemin du fichier sauvegardé
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrer et formater les données
    allocation = portfolio_df[portfolio_df['weight'] > 0.01].sort_values('weight', ascending=False).copy()
    
    # Convertir les poids en pourcentage
    allocation['weight'] = allocation['weight'] * 100
    
    # Renommer les colonnes pour plus de clarté
    allocation.rename(columns={
        'ticker': 'Symbole',
        'weight': 'Poids (%)',
        'score': 'Score'
    }, inplace=True)
    
    # Récupérer les métriques de performance s'il y en a
    metrics = {}
    for col in ['expected_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'effective_assets']:
        if col in portfolio_df.columns:
            # Convertir en pourcentage si nécessaire
            if col in ['expected_return', 'volatility', 'max_drawdown']:
                metrics[col] = f"{portfolio_df[col].iloc[0] * 100:.2f}%"
            else:
                metrics[col] = f"{portfolio_df[col].iloc[0]:.2f}"
    
    # Générer le fichier
    date_str = datetime.now().strftime('%Y%m%d')
    output_file = f"{output_dir}/portfolio_allocation_{method}_{date_str}.csv"
    
    # Sauvegarder
    allocation.to_csv(output_file, index=False, float_format='%.2f')
    
    # Ajouter les métriques dans un fichier texte
    metrics_file = f"{output_dir}/portfolio_metrics_{method}_{date_str}.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"Métriques de performance du portefeuille optimisé ({method}):\n")
        f.write("=" * 60 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.replace('_', ' ').title()}: {value}\n")
    
    logger.info(f"Allocation du portefeuille sauvegardée dans {output_file}")
    logger.info(f"Métriques de performance sauvegardées dans {metrics_file}")
    
    return output_file

def main():
    """Fonction principale."""
    # Parser les arguments de ligne de commande
    args = parse_args()
    
    # Charger la configuration
    config = load_config(args.config)
    
    # Charger les scores
    scores_df = load_scores_file(args.scores)
    
    # Sélectionner les N meilleures actions selon le score
    top_n_stocks = scores_df.sort_values('overall_score', ascending=False).head(args.num_stocks)
    tickers = top_n_stocks['ticker'].tolist()
    
    logger.info(f"Top {len(tickers)} actions sélectionnées pour l'optimisation: {', '.join(tickers)}")
    
    # Collecter les données de prix
    price_data = collect_price_data(tickers, args.lookback)
    
    # Initialiser l'optimiseur de portefeuille
    optimizer = PortfolioOptimizer(output_dir=args.output_dir)
    
    # Optimiser le portefeuille
    logger.info(f"Optimisation du portefeuille avec la méthode '{args.method}'...")
    
    optimized_portfolio = optimizer.optimize(
        price_data,
        top_n_stocks,
        method=args.method,
        risk_free_rate=args.risk_free_rate,
        max_weight=args.max_weight,
        min_weight=args.min_weight
    )
    
    # Afficher les résultats
    significant_weights = optimized_portfolio[optimized_portfolio['weight'] > args.min_weight]
    logger.info(f"\nPortefeuille optimisé ({args.method}):")
    for _, row in significant_weights.sort_values('weight', ascending=False).iterrows():
        logger.info(f"{row['ticker']}: {row['weight']*100:.2f}%")
    
    # Exporter les résultats
    export_portfolio_to_csv(optimized_portfolio, args.method, args.output_dir)
    
    # Générer le graphique d'allocation
    if args.plot:
        plot_portfolio_allocation(optimized_portfolio, args.method, args.output_dir)
    
    # Backtest du portefeuille si demandé
    if args.backtest:
        logger.info("\nBacktest du portefeuille optimisé...")
        
        # Extraire les poids du portefeuille
        weights = dict(zip(
            optimized_portfolio['ticker'],
            optimized_portfolio['weight']
        ))
        
        # Ne garder que les poids significatifs
        weights = {ticker: weight for ticker, weight in weights.items() if weight > args.min_weight}
        
        # Normaliser pour s'assurer que la somme est 1
        total_weight = sum(weights.values())
        weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
        
        # Période de backtest
        backtest_start = (datetime.now() - timedelta(days=365 * args.backtest_period)).strftime('%Y-%m-%d')
        
        # Exécuter le backtest
        backtest_results = optimizer.backtest_portfolio(
            price_data,
            weights,
            start_date=backtest_start,
            rebalance_frequency='M'  # Rééquilibrage mensuel
        )
        
        # Afficher les résultats du backtest
        backtest_metrics = {
            'Total Return': backtest_results['total_return'].iloc[0],
            'Annualized Return': backtest_results['annualized_return'].iloc[0],
            'Annualized Volatility': backtest_results['annualized_volatility'].iloc[0],
            'Sharpe Ratio': backtest_results['sharpe_ratio'].iloc[0],
            'Maximum Drawdown': backtest_results['max_drawdown'].iloc[0]
        }
        
        logger.info("\nRésultats du backtest:")
        for metric, value in backtest_metrics.items():
            if metric in ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown']:
                logger.info(f"{metric}: {value*100:.2f}%")
            else:
                logger.info(f"{metric}: {value:.2f}")
        
        # Générer le graphique du backtest
        if args.plot and '^GSPC' in price_data:
            benchmark = price_data['^GSPC']['Close']
            plot_backtest_results(backtest_results, benchmark, args.output_dir)
    
    logger.info("\nOptimisation terminée avec succès!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Erreur lors de l'exécution: {str(e)}")
        sys.exit(1)
