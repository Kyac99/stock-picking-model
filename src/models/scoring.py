"""
Module pour le scoring des actions basé sur des critères fondamentaux et techniques.
"""
import os
import logging
from typing import List, Dict, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configuration du logging
logger = logging.getLogger(__name__)

class StockScorer:
    """Classe de base pour tous les modèles de scoring."""
    
    def __init__(self, output_dir: str = "../data/results"):
        """
        Initialise un modèle de scoring.
        
        Args:
            output_dir: Répertoire pour sauvegarder les résultats
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_scores(self, scores: pd.DataFrame, filename: str) -> str:
        """
        Sauvegarde les scores dans un fichier CSV.
        
        Args:
            scores: DataFrame contenant les scores
            filename: Nom du fichier de sortie
            
        Returns:
            Chemin complet du fichier sauvegardé
        """
        filepath = os.path.join(self.output_dir, filename)
        scores.to_csv(filepath, index=True)
        logger.info(f"Scores sauvegardés dans {filepath}")
        return filepath


class FundamentalScorer(StockScorer):
    """Modèle de scoring basé sur les critères fondamentaux."""
    
    def __init__(
        self,
        output_dir: str = "../data/results",
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialise un modèle de scoring fondamental.
        
        Args:
            output_dir: Répertoire pour sauvegarder les résultats
            weights: Dictionnaire des poids pour chaque critère
        """
        super().__init__(output_dir)
        
        # Définir les poids par défaut si non spécifiés
        self.weights = weights or {
            # Valorisation
            'PE_Ratio': 0.10,
            'PB_Ratio': 0.05,
            'EV_EBITDA': 0.05,
            'Price_to_Sales': 0.05,
            
            # Rentabilité
            'ROE': 0.15,
            'ROA': 0.10,
            'ROIC': 0.15,
            'Profit_Margin': 0.10,
            
            # Croissance
            'Revenue_Growth': 0.10,
            'Earnings_Growth': 0.10,
            
            # Santé financière
            'Debt_to_Equity': 0.05,
            'Current_Ratio': 0.05,
            'Interest_Coverage': 0.05
        }
        
        # Normaliser les poids pour qu'ils somment à 1
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
            
        # Initialisations des scalers
        self.scalers = {}
        
    def score_stocks(
        self,
        fundamental_data: Dict[str, pd.DataFrame],
        save: bool = True
    ) -> pd.DataFrame:
        """
        Calcule les scores fondamentaux pour un ensemble d'actions.
        
        Args:
            fundamental_data: Dictionnaire de DataFrames avec les données fondamentales par ticker
            save: Si True, sauvegarde les scores
            
        Returns:
            DataFrame avec les scores par action
        """
        all_scores = []
        
        for ticker, data in fundamental_data.items():
            try:
                # Calculer les scores individuels
                scores = self._calculate_individual_scores(data)
                
                # Calculer le score global pondéré
                if scores:
                    weighted_score = self._calculate_weighted_score(scores)
                    
                    # Ajouter à la liste des scores
                    all_scores.append({
                        'ticker': ticker,
                        'overall_score': weighted_score,
                        **scores
                    })
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul des scores pour {ticker}: {str(e)}")
                
        # Créer un DataFrame avec tous les scores
        if not all_scores:
            logger.warning("Aucun score n'a pu être calculé")
            return pd.DataFrame()
            
        scores_df = pd.DataFrame(all_scores)
        
        # Trier par score global décroissant
        scores_df = scores_df.sort_values('overall_score', ascending=False)
        
        # Sauvegarder les scores
        if save:
            date_str = pd.Timestamp.now().strftime('%Y%m%d')
            self.save_scores(scores_df, f"fundamental_scores_{date_str}.csv")
            
        return scores_df
        
    def _calculate_individual_scores(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule les scores individuels pour chaque critère.
        
        Args:
            data: DataFrame contenant les données fondamentales d'une action
            
        Returns:
            Dictionnaire des scores par critère
        """
        scores = {}
        
        # --- Valorisation (plus bas est meilleur) ---
        # Price-to-Earnings (P/E)
        if 'PE_Ratio' in data.columns:
            scores['PE_Ratio'] = self._score_lower_better(data, 'PE_Ratio', 0, 50)
            
        # Price-to-Book (P/B)
        if 'PB_Ratio' in data.columns:
            scores['PB_Ratio'] = self._score_lower_better(data, 'PB_Ratio', 0, 10)
            
        # EV/EBITDA
        if 'EV_EBITDA' in data.columns:
            scores['EV_EBITDA'] = self._score_lower_better(data, 'EV_EBITDA', 0, 20)
            
        # Price-to-Sales
        if 'Price_to_Sales' in data.columns:
            scores['Price_to_Sales'] = self._score_lower_better(data, 'Price_to_Sales', 0, 10)
            
        # --- Rentabilité (plus haut est meilleur) ---
        # Return on Equity (ROE)
        if 'ROE' in data.columns:
            scores['ROE'] = self._score_higher_better(data, 'ROE', 0, 0.3)
            
        # Return on Assets (ROA)
        if 'ROA' in data.columns:
            scores['ROA'] = self._score_higher_better(data, 'ROA', 0, 0.15)
            
        # Return on Invested Capital (ROIC)
        if 'ROIC' in data.columns:
            scores['ROIC'] = self._score_higher_better(data, 'ROIC', 0, 0.2)
            
        # Profit Margin
        if 'Profit_Margin' in data.columns:
            scores['Profit_Margin'] = self._score_higher_better(data, 'Profit_Margin', 0, 0.3)
            
        # --- Croissance (plus haut est meilleur) ---
        # Revenue Growth
        if 'Revenue_Growth' in data.columns:
            scores['Revenue_Growth'] = self._score_higher_better(data, 'Revenue_Growth', 0, 0.25)
            
        # Earnings Growth
        if 'Earnings_Growth' in data.columns:
            scores['Earnings_Growth'] = self._score_higher_better(data, 'Earnings_Growth', 0, 0.3)
            
        # --- Santé financière ---
        # Debt-to-Equity (plus bas est meilleur)
        if 'Debt_to_Equity' in data.columns:
            scores['Debt_to_Equity'] = self._score_lower_better(data, 'Debt_to_Equity', 0, 3)
            
        # Current Ratio (plus haut est meilleur, mais pas trop)
        if 'Current_Ratio' in data.columns:
            scores['Current_Ratio'] = self._score_optimal_range(data, 'Current_Ratio', 1.5, 3.0, 0.5, 5.0)
            
        # Interest Coverage (plus haut est meilleur)
        if 'Interest_Coverage' in data.columns:
            scores['Interest_Coverage'] = self._score_higher_better(data, 'Interest_Coverage', 1, 10)
            
        return scores
        
    def _score_lower_better(
        self, 
        data: pd.DataFrame, 
        column: str, 
        min_value: float, 
        max_value: float
    ) -> float:
        """
        Calcule un score où les valeurs plus basses sont meilleures.
        
        Args:
            data: DataFrame contenant les données
            column: Nom de la colonne à évaluer
            min_value: Valeur minimale pour le scaling
            max_value: Valeur maximale pour le scaling
            
        Returns:
            Score entre 0 et 1
        """
        # Obtenir la dernière valeur
        value = data[column].iloc[-1]
        
        # Limiter aux valeurs min et max
        value = max(min_value, min(value, max_value))
        
        # Inverser pour que plus bas soit meilleur
        return 1 - ((value - min_value) / (max_value - min_value))
        
    def _score_higher_better(
        self, 
        data: pd.DataFrame, 
        column: str, 
        min_value: float, 
        max_value: float
    ) -> float:
        """
        Calcule un score où les valeurs plus hautes sont meilleures.
        
        Args:
            data: DataFrame contenant les données
            column: Nom de la colonne à évaluer
            min_value: Valeur minimale pour le scaling
            max_value: Valeur maximale pour le scaling
            
        Returns:
            Score entre 0 et 1
        """
        # Obtenir la dernière valeur
        value = data[column].iloc[-1]
        
        # Limiter aux valeurs min et max
        value = max(min_value, min(value, max_value))
        
        # Normaliser
        return (value - min_value) / (max_value - min_value)
        
    def _score_optimal_range(
        self, 
        data: pd.DataFrame, 
        column: str, 
        optimal_min: float, 
        optimal_max: float,
        absolute_min: float,
        absolute_max: float
    ) -> float:
        """
        Calcule un score où les valeurs dans une plage optimale sont meilleures.
        
        Args:
            data: DataFrame contenant les données
            column: Nom de la colonne à évaluer
            optimal_min: Valeur minimale de la plage optimale
            optimal_max: Valeur maximale de la plage optimale
            absolute_min: Valeur minimale absolue
            absolute_max: Valeur maximale absolue
            
        Returns:
            Score entre 0 et 1
        """
        # Obtenir la dernière valeur
        value = data[column].iloc[-1]
        
        # Limiter aux valeurs min et max absolues
        value = max(absolute_min, min(value, absolute_max))
        
        # Calculer le score selon la position par rapport à la plage optimale
        if optimal_min <= value <= optimal_max:
            # Dans la plage optimale = score maximal
            return 1.0
        elif value < optimal_min:
            # Sous la plage optimale
            return (value - absolute_min) / (optimal_min - absolute_min)
        else:
            # Au-dessus de la plage optimale
            return 1 - ((value - optimal_max) / (absolute_max - optimal_max))
            
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        Calcule un score global pondéré.
        
        Args:
            scores: Dictionnaire des scores par critère
            
        Returns:
            Score global pondéré entre 0 et 1
        """
        weighted_sum = 0
        total_applied_weight = 0
        
        for criterion, score in scores.items():
            if criterion in self.weights:
                weighted_sum += score * self.weights[criterion]
                total_applied_weight += self.weights[criterion]
                
        # Si aucun poids n'a été appliqué, retourner 0
        if total_applied_weight == 0:
            return 0
            
        # Normaliser par le poids total appliqué
        return weighted_sum / total_applied_weight


class TechnicalScorer(StockScorer):
    """Modèle de scoring basé sur les indicateurs techniques."""
    
    def __init__(
        self,
        output_dir: str = "../data/results",
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialise un modèle de scoring technique.
        
        Args:
            output_dir: Répertoire pour sauvegarder les résultats
            weights: Dictionnaire des poids pour chaque indicateur
        """
        super().__init__(output_dir)
        
        # Définir les poids par défaut si non spécifiés
        self.weights = weights or {
            # Indicateurs de tendance
            'trend_ma': 0.15,
            'macd': 0.10,
            
            # Indicateurs de momentum
            'rsi': 0.15,
            'stochastic': 0.10,
            
            # Indicateurs de volatilité
            'bollinger': 0.15,
            'atr': 0.05,
            
            # Indicateurs de volume
            'obv': 0.15,
            'volume_ma': 0.05,
            
            # Performance relative
            'relative_strength': 0.10
        }
        
        # Normaliser les poids pour qu'ils somment à 1
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
            
    def score_stocks(
        self,
        technical_data: Dict[str, pd.DataFrame],
        market_data: Optional[pd.DataFrame] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Calcule les scores techniques pour un ensemble d'actions.
        
        Args:
            technical_data: Dictionnaire de DataFrames avec les données techniques par ticker
            market_data: DataFrame avec les données de l'indice de référence (pour la force relative)
            save: Si True, sauvegarde les scores
            
        Returns:
            DataFrame avec les scores par action
        """
        all_scores = []
        
        for ticker, data in technical_data.items():
            try:
                # Calculer les scores individuels
                scores = self._calculate_individual_scores(data, market_data)
                
                # Calculer le score global pondéré
                if scores:
                    weighted_score = self._calculate_weighted_score(scores)
                    
                    # Ajouter à la liste des scores
                    all_scores.append({
                        'ticker': ticker,
                        'overall_score': weighted_score,
                        **scores
                    })
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul des scores techniques pour {ticker}: {str(e)}")
                
        # Créer un DataFrame avec tous les scores
        if not all_scores:
            logger.warning("Aucun score technique n'a pu être calculé")
            return pd.DataFrame()
            
        scores_df = pd.DataFrame(all_scores)
        
        # Trier par score global décroissant
        scores_df = scores_df.sort_values('overall_score', ascending=False)
        
        # Sauvegarder les scores
        if save:
            date_str = pd.Timestamp.now().strftime('%Y%m%d')
            self.save_scores(scores_df, f"technical_scores_{date_str}.csv")
            
        return scores_df
        
    def _calculate_individual_scores(
        self, 
        data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calcule les scores individuels pour chaque indicateur technique.
        
        Args:
            data: DataFrame contenant les données techniques d'une action
            market_data: DataFrame contenant les données de l'indice de référence
            
        Returns:
            Dictionnaire des scores par indicateur
        """
        scores = {}
        
        # S'assurer que nous avons suffisamment de données
        if len(data) < 200:
            logger.warning("Insuffisamment de données pour calculer les scores techniques")
            return scores
            
        # --- Indicateurs de tendance ---
        scores['trend_ma'] = self._score_moving_averages(data)
        scores['macd'] = self._score_macd(data)
        
        # --- Indicateurs de momentum ---
        scores['rsi'] = self._score_rsi(data)
        scores['stochastic'] = self._score_stochastic(data)
        
        # --- Indicateurs de volatilité ---
        scores['bollinger'] = self._score_bollinger_bands(data)
        scores['atr'] = self._score_atr(data)
        
        # --- Indicateurs de volume ---
        scores['obv'] = self._score_obv(data)
        scores['volume_ma'] = self._score_volume_ma(data)
        
        # --- Performance relative ---
        if market_data is not None:
            scores['relative_strength'] = self._score_relative_strength(data, market_data)
            
        return scores
        
    def _score_moving_averages(self, data: pd.DataFrame) -> float:
        """
        Calcule un score basé sur les moyennes mobiles.
        
        Args:
            data: DataFrame contenant les données techniques
            
        Returns:
            Score entre 0 et 1
        """
        # Vérifier la présence des colonnes nécessaires
        required_columns = ['Close', 'SMA_20', 'SMA_50', 'SMA_200']
        if not all(col in data.columns for col in required_columns):
            return 0.5  # Valeur neutre si données manquantes
            
        # Obtenir les dernières valeurs
        close = data['Close'].iloc[-1]
        sma20 = data['SMA_20'].iloc[-1]
        sma50 = data['SMA_50'].iloc[-1]
        sma200 = data['SMA_200'].iloc[-1]
        
        # Calculer le score sur différents critères
        score = 0
        
        # Prix au-dessus des moyennes mobiles
        if close > sma20:
            score += 0.2
        if close > sma50:
            score += 0.2
        if close > sma200:
            score += 0.2
            
        # Alignement des moyennes mobiles (Golden Cross)
        if sma20 > sma50 and sma50 > sma200:
            score += 0.4
        elif sma20 > sma50:
            score += 0.2
            
        return min(score, 1.0)
        
    def _score_macd(self, data: pd.DataFrame) -> float:
        """
        Calcule un score basé sur le MACD.
        
        Args:
            data: DataFrame contenant les données techniques
            
        Returns:
            Score entre 0 et 1
        """
        # Vérifier la présence des colonnes nécessaires
        required_columns = ['MACD_line', 'MACD_signal', 'MACD_histogram']
        if not all(col in data.columns for col in required_columns):
            return 0.5  # Valeur neutre si données manquantes
            
        # Obtenir les dernières valeurs
        macd_line = data['MACD_line'].iloc[-1]
        macd_signal = data['MACD_signal'].iloc[-1]
        macd_histogram = data['MACD_histogram'].iloc[-1]
        
        # Calculer le score
        score = 0
        
        # MACD au-dessus de sa ligne de signal
        if macd_line > macd_signal:
            score += 0.5
            
        # Histogramme positif et croissant
        if macd_histogram > 0:
            score += 0.3
            if len(data) >= 2 and macd_histogram > data['MACD_histogram'].iloc[-2]:
                score += 0.2
                
        return min(score, 1.0)
        
    def _score_rsi(self, data: pd.DataFrame) -> float:
        """
        Calcule un score basé sur le RSI.
        
        Args:
            data: DataFrame contenant les données techniques
            
        Returns:
            Score entre 0 et 1
        """
        # Vérifier la présence de la colonne nécessaire
        if 'RSI_14' not in data.columns:
            return 0.5  # Valeur neutre si données manquantes
            
        # Obtenir la dernière valeur
        rsi = data['RSI_14'].iloc[-1]
        
        # Calculer le score
        if rsi <= 30:
            # Survendu (0.7-1.0)
            return 0.7 + (30 - rsi) / 100
        elif rsi >= 70:
            # Suracheté (0.0-0.3)
            return 0.3 - (rsi - 70) / 100
        else:
            # Zone neutre (0.3-0.7), plus haut est meilleur
            return 0.3 + 0.4 * (rsi - 30) / 40
            
    def _score_stochastic(self, data: pd.DataFrame) -> float:
        """
        Calcule un score basé sur l'oscillateur stochastique.
        
        Args:
            data: DataFrame contenant les données techniques
            
        Returns:
            Score entre 0 et 1
        """
        # Vérifier la présence des colonnes nécessaires
        if '%K' not in data.columns or '%D' not in data.columns:
            return 0.5  # Valeur neutre si données manquantes
            
        # Obtenir les dernières valeurs
        k = data['%K'].iloc[-1]
        d = data['%D'].iloc[-1]
        
        # Calculer le score
        if k <= 20 and d <= 20:
            # Survendu (0.7-1.0)
            return 0.7 + (20 - min(k, d)) / 67
        elif k >= 80 and d >= 80:
            # Suracheté (0.0-0.3)
            return 0.3 - (max(k, d) - 80) / 67
        elif k > d:
            # Croisement haussier (0.6-0.7)
            return 0.6 + 0.1 * (k - d) / 20
        else:
            # Croisement baissier (0.3-0.4)
            return 0.4 - 0.1 * (d - k) / 20
            
    def _score_bollinger_bands(self, data: pd.DataFrame) -> float:
        """
        Calcule un score basé sur les Bandes de Bollinger.
        
        Args:
            data: DataFrame contenant les données techniques
            
        Returns:
            Score entre 0 et 1
        """
        # Vérifier la présence des colonnes nécessaires
        required_columns = ['Close', 'BB_upper', 'BB_middle', 'BB_lower']
        if not all(col in data.columns for col in required_columns):
            return 0.5  # Valeur neutre si données manquantes
            
        # Obtenir les dernières valeurs
        close = data['Close'].iloc[-1]
        bb_upper = data['BB_upper'].iloc[-1]
        bb_middle = data['BB_middle'].iloc[-1]
        bb_lower = data['BB_lower'].iloc[-1]
        
        # Calculer le score
        if close <= bb_lower:
            # Proche de la bande inférieure (signal d'achat)
            return 0.8 + 0.2 * (bb_lower - close) / (bb_lower * 0.05)
        elif close >= bb_upper:
            # Proche de la bande supérieure (signal de vente)
            return 0.2 - 0.2 * (close - bb_upper) / (bb_upper * 0.05)
        else:
            # Entre les bandes - score proportionnel à la position relative
            position = (close - bb_lower) / (bb_upper - bb_lower)
            # Convertir en score entre 0.2 et 0.8
            return 0.8 - 0.6 * position
            
    def _score_atr(self, data: pd.DataFrame) -> float:
        """
        Calcule un score basé sur l'Average True Range (ATR).
        
        Args:
            data: DataFrame contenant les données techniques
            
        Returns:
            Score entre 0 et 1
        """
        # Vérifier la présence des colonnes nécessaires
        if 'ATR_14' not in data.columns or 'Close' not in data.columns:
            return 0.5  # Valeur neutre si données manquantes
            
        # Obtenir les dernières valeurs
        atr = data['ATR_14'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        # Calculer l'ATR en pourcentage du prix
        atr_pct = atr / close * 100
        
        # Calculer le score - une volatilité modérée est préférable
        if atr_pct <= 1:
            # Très faible volatilité
            return 0.4 + 0.2 * atr_pct
        elif atr_pct <= 3:
            # Volatilité idéale
            return 0.6 + 0.4 * (3 - atr_pct) / 2
        else:
            # Volatilité élevée
            return max(0.1, 0.6 - 0.1 * (atr_pct - 3))
            
    def _score_obv(self, data: pd.DataFrame) -> float:
        """
        Calcule un score basé sur l'On-Balance Volume (OBV).
        
        Args:
            data: DataFrame contenant les données techniques
            
        Returns:
            Score entre 0 et 1
        """
        # Vérifier la présence de la colonne nécessaire
        if 'OBV' not in data.columns:
            return 0.5  # Valeur neutre si données manquantes
            
        # Calculer la tendance de l'OBV sur les 20 derniers jours
        obv_values = data['OBV'].iloc[-20:].values
        
        # Tendance linéaire simple
        x = np.arange(len(obv_values))
        slope, _ = np.polyfit(x, obv_values, 1)
        
        # Normaliser la pente
        normalized_slope = min(max(slope / (np.mean(np.abs(obv_values)) * 0.02), -1), 1)
        
        # Convertir en score (pente positive = bon score)
        return 0.5 + 0.5 * normalized_slope
        
    def _score_volume_ma(self, data: pd.DataFrame) -> float:
        """
        Calcule un score basé sur la moyenne mobile du volume.
        
        Args:
            data: DataFrame contenant les données techniques
            
        Returns:
            Score entre 0 et 1
        """
        # Vérifier la présence des colonnes nécessaires
        if 'Volume' not in data.columns or 'Volume_MA_20' not in data.columns:
            return 0.5  # Valeur neutre si données manquantes
            
        # Obtenir les dernières valeurs
        volume = data['Volume'].iloc[-1]
        volume_ma = data['Volume_MA_20'].iloc[-1]
        
        # Calculer le ratio volume / moyenne mobile
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1
        
        # Calculer le score
        if volume_ratio <= 0.5:
            # Volume très faible
            return 0.3
        elif volume_ratio <= 1:
            # Volume inférieur à la moyenne
            return 0.3 + 0.2 * volume_ratio
        elif volume_ratio <= 2:
            # Volume supérieur à la moyenne (bon signe)
            return 0.5 + 0.4 * (volume_ratio - 1)
        else:
            # Volume très élevé (potentiellement excessif)
            return max(0.5, 0.9 - 0.1 * (volume_ratio - 2))
            
    def _score_relative_strength(
        self, 
        data: pd.DataFrame, 
        market_data: pd.DataFrame
    ) -> float:
        """
        Calcule un score basé sur la force relative par rapport au marché.
        
        Args:
            data: DataFrame contenant les données techniques de l'action
            market_data: DataFrame contenant les données de l'indice de référence
            
        Returns:
            Score entre 0 et 1
        """
        # Vérifier la présence des colonnes nécessaires
        if 'Close' not in data.columns or 'Close' not in market_data.columns:
            return 0.5  # Valeur neutre si données manquantes
            
        # S'assurer que les index sont des dates
        stock_data = data.copy()
        market_data_aligned = market_data.copy()
        
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)
        if not isinstance(market_data_aligned.index, pd.DatetimeIndex):
            market_data_aligned.index = pd.to_datetime(market_data_aligned.index)
            
        # Aligner les données sur les mêmes dates
        common_dates = stock_data.index.intersection(market_data_aligned.index)
        if len(common_dates) < 60:  # Au moins 60 jours de données
            return 0.5
            
        stock_data = stock_data.loc[common_dates]
        market_data_aligned = market_data_aligned.loc[common_dates]
        
        # Calculer les rendements
        stock_returns = stock_data['Close'].pct_change().dropna()
        market_returns = market_data_aligned['Close'].pct_change().dropna()
        
        # Calculer la force relative à différentes périodes
        periods = [5, 20, 60]  # 1 semaine, 1 mois, 3 mois
        rs_scores = []
        
        for period in periods:
            if len(stock_returns) < period:
                continue
                
            # Rendement cumulé
            stock_cum_return = (1 + stock_returns.iloc[-period:]).prod() - 1
            market_cum_return = (1 + market_returns.iloc[-period:]).prod() - 1
            
            # Force relative (différence de rendement)
            relative_strength = stock_cum_return - market_cum_return
            
            # Convertir en score entre 0 et 1
            if relative_strength <= -0.2:  # Sous-performance forte
                rs_score = 0.0
            elif relative_strength <= 0:  # Sous-performance légère
                rs_score = 0.5 + 2.5 * relative_strength  # 0.5 à 0.0
            elif relative_strength <= 0.2:  # Surperformance légère à moyenne
                rs_score = 0.5 + 2.5 * relative_strength  # 0.5 à 1.0
            else:  # Surperformance forte
                rs_score = 1.0
                
            rs_scores.append(rs_score)
            
        # Si aucun score n'a pu être calculé, retourner une valeur neutre
        if not rs_scores:
            return 0.5
            
        # Moyenne pondérée des scores par période (plus de poids aux périodes récentes)
        weights = [0.5, 0.3, 0.2]  # 5j: 50%, 20j: 30%, 60j: 20%
        return sum(score * weight for score, weight in zip(rs_scores, weights[:len(rs_scores)])) / sum(weights[:len(rs_scores)])
        
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        Calcule un score global pondéré.
        
        Args:
            scores: Dictionnaire des scores par indicateur
            
        Returns:
            Score global pondéré entre 0 et 1
        """
        weighted_sum = 0
        total_applied_weight = 0
        
        for indicator, score in scores.items():
            if indicator in self.weights:
                weighted_sum += score * self.weights[indicator]
                total_applied_weight += self.weights[indicator]
                
        # Si aucun poids n'a été appliqué, retourner 0.5 (valeur neutre)
        if total_applied_weight == 0:
            return 0.5
            
        # Normaliser par le poids total appliqué
        return weighted_sum / total_applied_weight


class MultifactorScorer(StockScorer):
    """Modèle de scoring combinant des critères fondamentaux et techniques."""
    
    def __init__(
        self,
        output_dir: str = "../data/results",
        fundamental_weight: float = 0.7,
        technical_weight: float = 0.3,
        fundamental_scorer: Optional[FundamentalScorer] = None,
        technical_scorer: Optional[TechnicalScorer] = None
    ):
        """
        Initialise un modèle de scoring multifactoriel.
        
        Args:
            output_dir: Répertoire pour sauvegarder les résultats
            fundamental_weight: Poids pour le score fondamental (0-1)
            technical_weight: Poids pour le score technique (0-1)
            fundamental_scorer: Instance de FundamentalScorer à utiliser
            technical_scorer: Instance de TechnicalScorer à utiliser
        """
        super().__init__(output_dir)
        
        # Normaliser les poids pour qu'ils somment à 1
        total_weight = fundamental_weight + technical_weight
        self.fundamental_weight = fundamental_weight / total_weight
        self.technical_weight = technical_weight / total_weight
        
        # Initialiser les scorers si non fournis
        self.fundamental_scorer = fundamental_scorer or FundamentalScorer(output_dir)
        self.technical_scorer = technical_scorer or TechnicalScorer(output_dir)
        
    def score_stocks(
        self,
        fundamental_data: Dict[str, pd.DataFrame],
        technical_data: Dict[str, pd.DataFrame],
        market_data: Optional[pd.DataFrame] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Calcule les scores combinés pour un ensemble d'actions.
        
        Args:
            fundamental_data: Dictionnaire de DataFrames avec les données fondamentales par ticker
            technical_data: Dictionnaire de DataFrames avec les données techniques par ticker
            market_data: DataFrame avec les données de l'indice de référence (pour la force relative)
            save: Si True, sauvegarde les scores
            
        Returns:
            DataFrame avec les scores par action
        """
        # Calculer les scores fondamentaux et techniques séparément
        # save=False pour éviter de sauvegarder les scores intermédiaires
        fundamental_scores = self.fundamental_scorer.score_stocks(fundamental_data, save=False)
        technical_scores = self.technical_scorer.score_stocks(technical_data, market_data, save=False)
        
        # Vérifier si des scores ont pu être calculés
        if fundamental_scores.empty and technical_scores.empty:
            logger.warning("Aucun score n'a pu être calculé")
            return pd.DataFrame()
            
        # Fusionner les scores
        combined_scores = self._combine_scores(fundamental_scores, technical_scores)
        
        # Sauvegarder les scores
        if save:
            date_str = pd.Timestamp.now().strftime('%Y%m%d')
            self.save_scores(combined_scores, f"multifactor_scores_{date_str}.csv")
            
        return combined_scores
        
    def _combine_scores(
        self,
        fundamental_scores: pd.DataFrame,
        technical_scores: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine les scores fondamentaux et techniques.
        
        Args:
            fundamental_scores: DataFrame avec les scores fondamentaux
            technical_scores: DataFrame avec les scores techniques
            
        Returns:
            DataFrame combinant les scores fondamentaux et techniques
        """
        # Gérer le cas où l'un des DataFrames est vide
        if fundamental_scores.empty:
            logger.warning("Scores fondamentaux manquants, utilisation uniquement des scores techniques")
            technical_scores['overall_score'] = technical_scores['overall_score']
            return technical_scores
            
        if technical_scores.empty:
            logger.warning("Scores techniques manquants, utilisation uniquement des scores fondamentaux")
            fundamental_scores['overall_score'] = fundamental_scores['overall_score']
            return fundamental_scores
            
        # Fusionner sur le ticker
        combined = pd.merge(
            fundamental_scores[['ticker', 'overall_score']],
            technical_scores[['ticker', 'overall_score']],
            on='ticker',
            how='outer',
            suffixes=('_fundamental', '_technical')
        )
        
        # Gérer les valeurs manquantes
        combined.fillna({
            'overall_score_fundamental': 0.5,  # Valeur neutre pour les scores fondamentaux manquants
            'overall_score_technical': 0.5     # Valeur neutre pour les scores techniques manquants
        }, inplace=True)
        
        # Calculer le score global pondéré
        combined['overall_score'] = (
            combined['overall_score_fundamental'] * self.fundamental_weight +
            combined['overall_score_technical'] * self.technical_weight
        )
        
        # Trier par score global décroissant
        combined = combined.sort_values('overall_score', ascending=False)
        
        return combined
