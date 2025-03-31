"""
Module pour l'optimisation de portefeuille basée sur les résultats du modèle de stock picking.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy.optimize import minimize

# Configuration du logging
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Classe pour l'optimisation de portefeuille basée sur les scores du modèle."""
    
    def __init__(self, output_dir: str = "../data/results"):
        """
        Initialise un optimiseur de portefeuille.
        
        Args:
            output_dir: Répertoire pour sauvegarder les résultats
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def optimize(
        self,
        price_data: Dict[str, pd.DataFrame],
        scores: pd.DataFrame,
        method: str = "markowitz",
        risk_free_rate: float = 0.02,
        risk_aversion: float = 0.5,
        target_return: Optional[float] = None,
        max_weight: float = 0.25,
        min_weight: float = 0.01,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Optimise un portefeuille basé sur les données de prix historiques et les scores du modèle.
        
        Args:
            price_data: Dictionnaire des DataFrames de prix par ticker
            scores: DataFrame contenant les scores par ticker
            method: Méthode d'optimisation ('markowitz', 'score_based', 'risk_parity')
            risk_free_rate: Taux sans risque annualisé
            risk_aversion: Coefficient d'aversion au risque (0-1)
            target_return: Rendement cible annualisé (si None, maximise le ratio de Sharpe)
            max_weight: Poids maximum par actif
            min_weight: Poids minimum par actif (si inclus dans le portefeuille)
            save: Si True, sauvegarde les résultats
            
        Returns:
            DataFrame contenant les poids optimaux et les métriques de performance
        """
        # Filtrer les tickers pour lesquels nous avons des données de prix et des scores
        valid_tickers = [ticker for ticker in scores['ticker'] if ticker in price_data]
        
        if not valid_tickers:
            logger.error("Aucun ticker valide trouvé avec des données de prix et des scores")
            return pd.DataFrame()
            
        # Filtrer les scores pour ne garder que les tickers valides
        filtered_scores = scores[scores['ticker'].isin(valid_tickers)].copy()
        
        # Calculer les rendements et la matrice de covariance
        returns_data, cov_matrix = self._calculate_returns_and_covariance(
            price_data, valid_tickers
        )
        
        # Sélectionner la méthode d'optimisation
        if method == "markowitz":
            optimal_weights = self._markowitz_optimization(
                returns_data, cov_matrix, 
                risk_free_rate, risk_aversion, target_return,
                max_weight, min_weight
            )
        elif method == "score_based":
            optimal_weights = self._score_based_optimization(
                filtered_scores, cov_matrix,
                max_weight, min_weight
            )
        elif method == "risk_parity":
            optimal_weights = self._risk_parity_optimization(
                cov_matrix, max_weight, min_weight
            )
        else:
            logger.error(f"Méthode d'optimisation inconnue: {method}")
            return pd.DataFrame()
            
        # Calculer les métriques de performance
        performance_metrics = self._calculate_performance_metrics(
            optimal_weights, returns_data, cov_matrix, risk_free_rate
        )
        
        # Créer un DataFrame avec les résultats
        portfolio_df = pd.DataFrame({
            'ticker': valid_tickers,
            'weight': [optimal_weights.get(ticker, 0) for ticker in valid_tickers],
            'score': [filtered_scores.loc[filtered_scores['ticker'] == ticker, 'overall_score'].iloc[0] 
                      if ticker in filtered_scores['ticker'].values else np.nan 
                      for ticker in valid_tickers]
        })
        
        # Trier par poids décroissant
        portfolio_df = portfolio_df.sort_values('weight', ascending=False)
        
        # Ajouter les métriques de performance
        for metric, value in performance_metrics.items():
            portfolio_df.loc[0, metric] = value
            
        # Sauvegarder les résultats
        if save:
            date_str = pd.Timestamp.now().strftime('%Y%m%d')
            output_file = f"optimal_portfolio_{method}_{date_str}.csv"
            output_path = os.path.join(self.output_dir, output_file)
            portfolio_df.to_csv(output_path, index=False)
            logger.info(f"Résultats de l'optimisation sauvegardés dans {output_path}")
            
        return portfolio_df
        
    def _calculate_returns_and_covariance(
        self, 
        price_data: Dict[str, pd.DataFrame],
        tickers: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcule les rendements historiques et la matrice de covariance.
        
        Args:
            price_data: Dictionnaire des DataFrames de prix par ticker
            tickers: Liste des tickers à inclure
            
        Returns:
            Tuple de (DataFrame des rendements, matrice de covariance)
        """
        # Extraire les prix de clôture pour chaque ticker
        closes = pd.DataFrame()
        
        for ticker in tickers:
            if ticker in price_data:
                closes[ticker] = price_data[ticker]['Close']
                
        # Calculer les rendements journaliers
        returns = closes.pct_change().dropna()
        
        # Calculer la matrice de covariance
        cov_matrix = returns.cov()
        
        return returns, cov_matrix
        
    def _markowitz_optimization(
        self,
        returns_data: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
        risk_aversion: float,
        target_return: Optional[float],
        max_weight: float,
        min_weight: float
    ) -> Dict[str, float]:
        """
        Effectue une optimisation de portefeuille selon le modèle de Markowitz.
        
        Args:
            returns_data: DataFrame des rendements historiques
            cov_matrix: Matrice de covariance des rendements
            risk_free_rate: Taux sans risque annualisé
            risk_aversion: Coefficient d'aversion au risque (0-1)
            target_return: Rendement cible annualisé
            max_weight: Poids maximum par actif
            min_weight: Poids minimum par actif
            
        Returns:
            Dictionnaire des poids optimaux par ticker
        """
        tickers = returns_data.columns
        n_assets = len(tickers)
        
        # Calculer les rendements moyens
        mean_returns = returns_data.mean()
        
        # Convertir en rendements annualisés (approximation simple)
        annual_returns = mean_returns * 252
        annual_cov = cov_matrix * 252
        
        # Définir la fonction objectif selon le cas
        if target_return is not None:
            # Minimisation de la variance avec contrainte de rendement
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(annual_cov, weights))
                return portfolio_variance
                
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Somme des poids = 1
                {'type': 'eq', 'fun': lambda x: np.sum(x * annual_returns) - target_return}  # Rendement cible
            ]
        else:
            # Maximisation du ratio de Sharpe
            def objective(weights):
                portfolio_return = np.sum(weights * annual_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                return -sharpe_ratio  # Négatif car on minimise
                
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Somme des poids = 1
            ]
            
        # Bornes pour les poids
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Optimisation
        initial_weights = np.ones(n_assets) / n_assets  # Poids initiaux égaux
        
        try:
            result = minimize(
                objective, 
                initial_weights, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = {ticker: weight for ticker, weight in zip(tickers, result.x)}
                return optimal_weights
            else:
                logger.error(f"L'optimisation a échoué: {result.message}")
                return {ticker: 1/n_assets for ticker in tickers}  # Poids égaux en cas d'échec
                
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation: {str(e)}")
            return {ticker: 1/n_assets for ticker in tickers}  # Poids égaux en cas d'erreur
            
    def _score_based_optimization(
        self,
        scores: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        max_weight: float,
        min_weight: float
    ) -> Dict[str, float]:
        """
        Effectue une optimisation basée sur les scores du modèle.
        
        Args:
            scores: DataFrame contenant les scores par ticker
            cov_matrix: Matrice de covariance des rendements
            max_weight: Poids maximum par actif
            min_weight: Poids minimum par actif
            
        Returns:
            Dictionnaire des poids optimaux par ticker
        """
        tickers = cov_matrix.columns
        n_assets = len(tickers)
        
        # Extraire les scores
        ticker_scores = {}
        for _, row in scores.iterrows():
            ticker = row['ticker']
            if ticker in tickers:
                ticker_scores[ticker] = row['overall_score']
                
        # Normaliser les scores
        total_score = sum(ticker_scores.values())
        normalized_scores = {ticker: score / total_score for ticker, score in ticker_scores.items()}
        
        # Fonction d'optimisation : maximiser le score pondéré tout en minimisant la variance
        def objective(weights):
            # Poids du score dans l'objectif (paramétrable)
            score_weight = 0.7
            risk_weight = 0.3
            
            # Calcul du score du portefeuille
            portfolio_score = sum(weights[i] * normalized_scores.get(ticker, 0) 
                                  for i, ticker in enumerate(tickers))
            
            # Calcul de la variance du portefeuille
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Normalisation de la variance
            max_var = np.max(np.diag(cov_matrix))
            normalized_variance = portfolio_variance / max_var
            
            # Fonction objectif combinée : maximiser le score, minimiser la variance
            return -(score_weight * portfolio_score - risk_weight * normalized_variance)
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Somme des poids = 1
        ]
        
        # Bornes pour les poids
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Optimisation
        initial_weights = np.array([normalized_scores.get(ticker, 1/n_assets) for ticker in tickers])
        initial_weights = initial_weights / sum(initial_weights)  # Normaliser
        
        try:
            result = minimize(
                objective, 
                initial_weights, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = {ticker: weight for ticker, weight in zip(tickers, result.x)}
                return optimal_weights
            else:
                logger.error(f"L'optimisation basée sur les scores a échoué: {result.message}")
                return {ticker: normalized_scores.get(ticker, 1/n_assets) for ticker in tickers}
                
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation basée sur les scores: {str(e)}")
            return {ticker: normalized_scores.get(ticker, 1/n_assets) for ticker in tickers}
            
    def _risk_parity_optimization(
        self,
        cov_matrix: pd.DataFrame,
        max_weight: float,
        min_weight: float
    ) -> Dict[str, float]:
        """
        Effectue une optimisation par parité de risque.
        
        Args:
            cov_matrix: Matrice de covariance des rendements
            max_weight: Poids maximum par actif
            min_weight: Poids minimum par actif
            
        Returns:
            Dictionnaire des poids optimaux par ticker
        """
        tickers = cov_matrix.columns
        n_assets = len(tickers)
        
        # Fonction objectif pour la parité de risque
        def risk_parity_objective(weights):
            # Convertir les poids en array NumPy
            weights = np.array(weights)
            
            # Normaliser les poids pour s'assurer que la somme est 1
            weights = weights / np.sum(weights)
            
            # Calculer les contributions au risque pour chaque actif
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            marginal_contributions = np.dot(cov_matrix, weights) / portfolio_volatility
            risk_contributions = weights * marginal_contributions
            
            # Objectif : minimiser la somme des écarts carrés entre les contributions au risque
            target_risk_contribution = portfolio_volatility / n_assets
            risk_diffs = risk_contributions - target_risk_contribution
            
            return np.sum(risk_diffs**2)
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Somme des poids = 1
        ]
        
        # Bornes pour les poids
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Optimisation
        initial_weights = np.ones(n_assets) / n_assets  # Poids initiaux égaux
        
        try:
            result = minimize(
                risk_parity_objective, 
                initial_weights, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'ftol': 1e-12, 'maxiter': 1000}
            )
            
            if result.success:
                # Normaliser les poids pour s'assurer que la somme est 1
                weights = result.x / np.sum(result.x)
                optimal_weights = {ticker: weight for ticker, weight in zip(tickers, weights)}
                return optimal_weights
            else:
                logger.error(f"L'optimisation par parité de risque a échoué: {result.message}")
                return {ticker: 1/n_assets for ticker in tickers}  # Poids égaux en cas d'échec
                
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation par parité de risque: {str(e)}")
            return {ticker: 1/n_assets for ticker in tickers}  # Poids égaux en cas d'erreur
            
    def _calculate_performance_metrics(
        self,
        weights: Dict[str, float],
        returns_data: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float
    ) -> Dict[str, float]:
        """
        Calcule les métriques de performance du portefeuille optimisé.
        
        Args:
            weights: Dictionnaire des poids optimaux par ticker
            returns_data: DataFrame des rendements historiques
            cov_matrix: Matrice de covariance des rendements
            risk_free_rate: Taux sans risque annualisé
            
        Returns:
            Dictionnaire des métriques de performance
        """
        # Convertir les poids en array NumPy dans l'ordre des tickers de returns_data
        tickers = returns_data.columns
        weights_array = np.array([weights.get(ticker, 0) for ticker in tickers])
        
        # Calculer les rendements moyens
        mean_returns = returns_data.mean()
        
        # Convertir en rendements annualisés
        annual_returns = mean_returns * 252
        annual_cov = cov_matrix * 252
        
        # Rendement attendu du portefeuille
        portfolio_return = np.sum(weights_array * annual_returns)
        
        # Volatilité (risque) du portefeuille
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(annual_cov, weights_array)))
        
        # Ratio de Sharpe
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Calculer le drawdown maximum historique
        portfolio_returns = np.sum(returns_data * weights_array, axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculer la diversification (nombre effectif d'actifs)
        non_zero_weights = weights_array[weights_array > 0.001]  # Ignorer les poids très petits
        herfindahl_index = np.sum(non_zero_weights**2)
        effective_n = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        # Métriques de performance
        performance_metrics = {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'effective_assets': effective_n
        }
        
        return performance_metrics
    
    def generate_efficient_frontier(
        self,
        price_data: Dict[str, pd.DataFrame],
        tickers: List[str],
        n_points: int = 20,
        risk_free_rate: float = 0.02,
        max_weight: float = 0.25,
        min_weight: float = 0.01,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Génère la frontière efficiente pour un ensemble d'actifs.
        
        Args:
            price_data: Dictionnaire des DataFrames de prix par ticker
            tickers: Liste des tickers à inclure
            n_points: Nombre de points sur la frontière efficiente
            risk_free_rate: Taux sans risque annualisé
            max_weight: Poids maximum par actif
            min_weight: Poids minimum par actif
            save: Si True, sauvegarde les résultats
            
        Returns:
            DataFrame contenant les points de la frontière efficiente
        """
        # Calculer les rendements et la matrice de covariance
        returns_data, cov_matrix = self._calculate_returns_and_covariance(
            price_data, tickers
        )
        
        # Calculer les rendements moyens
        mean_returns = returns_data.mean()
        
        # Convertir en rendements annualisés
        annual_returns = mean_returns * 252
        annual_cov = cov_matrix * 252
        
        # Trouver les portefeuilles à variance minimale et à rendement maximal
        min_volatility_weights = self._markowitz_optimization(
            returns_data, cov_matrix, 
            risk_free_rate, risk_aversion=0.5, target_return=None,
            max_weight=max_weight, min_weight=min_weight
        )
        
        # Calculer le rendement du portefeuille à variance minimale
        min_vol_return = sum(annual_returns[ticker] * weight for ticker, weight in min_volatility_weights.items())
        
        # Estimer le rendement maximum atteignable
        max_return_asset = annual_returns.idxmax()
        max_return = annual_returns[max_return_asset]
        
        # Générer des rendements cibles entre min_vol_return et max_return
        target_returns = np.linspace(min_vol_return, max_return, n_points)
        
        # Calculer les portefeuilles optimaux pour chaque rendement cible
        efficient_frontier = []
        
        for target_return in target_returns:
            weights = self._markowitz_optimization(
                returns_data, cov_matrix, 
                risk_free_rate, risk_aversion=0.5, target_return=target_return,
                max_weight=max_weight, min_weight=min_weight
            )
            
            # Calculer les métriques de performance
            metrics = self._calculate_performance_metrics(
                weights, returns_data, cov_matrix, risk_free_rate
            )
            
            # Stocker les résultats
            point = {
                'target_return': target_return,
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                **{f'weight_{ticker}': weights.get(ticker, 0) for ticker in tickers}
            }
            
            efficient_frontier.append(point)
            
        # Créer un DataFrame avec les résultats
        frontier_df = pd.DataFrame(efficient_frontier)
        
        # Sauvegarder les résultats
        if save:
            date_str = pd.Timestamp.now().strftime('%Y%m%d')
            output_file = f"efficient_frontier_{date_str}.csv"
            output_path = os.path.join(self.output_dir, output_file)
            frontier_df.to_csv(output_path, index=False)
            logger.info(f"Frontière efficiente sauvegardée dans {output_path}")
            
        return frontier_df
        
    def backtest_portfolio(
        self,
        price_data: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_frequency: str = 'M',  # Monthly
        save: bool = True
    ) -> pd.DataFrame:
        """
        Effectue un backtest du portefeuille avec les poids spécifiés.
        
        Args:
            price_data: Dictionnaire des DataFrames de prix par ticker
            weights: Dictionnaire des poids par ticker
            start_date: Date de début du backtest (format YYYY-MM-DD)
            end_date: Date de fin du backtest (format YYYY-MM-DD)
            rebalance_frequency: Fréquence de rééquilibrage ('D', 'W', 'M', 'Q', 'Y')
            save: Si True, sauvegarde les résultats
            
        Returns:
            DataFrame contenant les résultats du backtest
        """
        # Extraire les tickers du portefeuille
        tickers = list(weights.keys())
        
        # Créer un DataFrame de prix uniformisé
        prices = pd.DataFrame()
        
        for ticker in tickers:
            if ticker in price_data:
                prices[ticker] = price_data[ticker]['Close']
                
        # Filtrer par dates si spécifiées
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
            
        # Calculer les rendements
        returns = prices.pct_change().dropna()
        
        # Initialiser les valeurs du portefeuille
        portfolio_values = pd.DataFrame(index=returns.index)
        portfolio_values['portfolio_value'] = 0
        
        # Définir les dates de rééquilibrage
        if rebalance_frequency == 'D':
            rebalance_dates = returns.index
        else:
            rebalance_dates = pd.date_range(
                start=returns.index[0], 
                end=returns.index[-1], 
                freq=rebalance_frequency
            )
            # S'assurer que les dates de rééquilibrage sont des jours de trading
            rebalance_dates = [date for date in rebalance_dates if date in returns.index]
            
        # Simuler la performance du portefeuille
        current_weights = {ticker: weight for ticker, weight in weights.items()}
        portfolio_value = 1.0  # Commencer avec une valeur normalisée de 1
        
        for i, date in enumerate(returns.index):
            # Mettre à jour la valeur du portefeuille avec les rendements de la journée
            daily_return = sum(current_weights.get(ticker, 0) * returns.loc[date, ticker] 
                               for ticker in tickers if ticker in returns.columns)
            
            portfolio_value *= (1 + daily_return)
            portfolio_values.loc[date, 'portfolio_value'] = portfolio_value
            
            # Rééquilibrer si c'est une date de rééquilibrage (sauf le dernier jour)
            if date in rebalance_dates and i < len(returns.index) - 1:
                # Mettre à jour les poids avec les rendements depuis le dernier rééquilibrage
                current_weights = {ticker: weight for ticker, weight in weights.items()}
                
        # Calculer les métriques de performance
        portfolio_returns = portfolio_values['portfolio_value'].pct_change().dropna()
        
        # Rendement annualisé
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        annualized_return = (portfolio_value ** (365 / days)) - 1 if days > 0 else 0
        
        # Volatilité annualisée
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Ratio de Sharpe
        risk_free_rate = 0.02  # Taux sans risque annualisé
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Ajouter les métriques au DataFrame
        portfolio_values['cumulative_return'] = portfolio_values['portfolio_value'] / portfolio_values['portfolio_value'].iloc[0] - 1
        portfolio_values['drawdown'] = (portfolio_values['portfolio_value'] / portfolio_values['portfolio_value'].cummax()) - 1
        
        # Ajouter les métriques de performance
        performance_metrics = {
            'start_date': portfolio_values.index[0],
            'end_date': portfolio_values.index[-1],
            'total_return': portfolio_value - 1,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        # Sauvegarder les résultats
        if save:
            date_str = pd.Timestamp.now().strftime('%Y%m%d')
            output_file = f"backtest_results_{date_str}.csv"
            output_path = os.path.join(self.output_dir, output_file)
            portfolio_values.to_csv(output_path, index=True)
            
            # Sauvegarder aussi les métriques de performance
            metrics_df = pd.DataFrame([performance_metrics])
            metrics_file = f"backtest_metrics_{date_str}.csv"
            metrics_path = os.path.join(self.output_dir, metrics_file)
            metrics_df.to_csv(metrics_path, index=False)
            
            logger.info(f"Résultats du backtest sauvegardés dans {output_path}")
            
        # Ajouter les métriques dans le DataFrame de retour
        for metric, value in performance_metrics.items():
            if metric not in ['start_date', 'end_date']:
                portfolio_values.loc[portfolio_values.index[0], metric] = value
            
        return portfolio_values
