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
        
        # Calculer les rendements pour l'action et l'indice
        stock_returns = stock_data['Close'].pct_change()
        market_returns = market_data_aligned['Close'].pct_change()
        
        # Calculer la force relative sur différentes périodes
        # 1 mois (environ 21 jours de trading)
        rs_1m = (1 + stock_returns.iloc[-21:]).prod() / (1 + market_returns.iloc[-21:]).prod()
        
        # 3 mois (environ 63 jours de trading)
        rs_3m = (1 + stock_returns.iloc[-63:]).prod() / (1 + market_returns.iloc[-63:]).prod()
        
        # 6 mois (environ 126 jours de trading)
        rs_6m = (1 + stock_returns.iloc[-126:]).prod() / (1 + market_returns.iloc[-126:]).prod()
        
        # Pondérer les périodes (privilégier les plus récentes)
        weighted_rs = 0.5 * rs_1m + 0.3 * rs_3m + 0.2 * rs_6m
        
        # Convertir en score entre 0 et 1
        # Une force relative de 1 signifie une performance identique au marché
        if weighted_rs >= 1:
            # Surperformance - score entre 0.5 et 1.0
            return min(0.5 + 0.5 * min(weighted_rs - 1, 1), 1.0)
        else:
            # Sous-performance - score entre 0 et 0.5
            return max(0.5 - 0.5 * min(1 - weighted_rs, 1), 0.0)
    
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
    """Modèle de scoring combinant des critères fondamentaux, techniques et qualitatifs."""
    
    def __init__(
        self,
        output_dir: str = "../data/results",
        factor_weights: Optional[Dict[str, float]] = None,
        quality_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialise un modèle de scoring multifactoriel.
        
        Args:
            output_dir: Répertoire pour sauvegarder les résultats
            factor_weights: Dictionnaire des poids pour chaque facteur
            quality_weights: Dictionnaire des poids pour les critères qualitatifs
        """
        super().__init__(output_dir)
        
        # Définir les poids par défaut si non spécifiés
        self.factor_weights = factor_weights or {
            'fundamental': 0.6,
            'technical': 0.3,
            'quality': 0.1
        }
        
        # Normaliser les poids des facteurs
        total_weight = sum(self.factor_weights.values())
        for key in self.factor_weights:
            self.factor_weights[key] /= total_weight
            
        # Définir les poids des critères qualitatifs
        self.quality_weights = quality_weights or {
            'management': 0.3,
            'industry_outlook': 0.3,
            'competitive_advantage': 0.2,
            'esg_score': 0.1,
            'regulatory_risks': 0.1
        }
        
        # Normaliser les poids des critères qualitatifs
        quality_total = sum(self.quality_weights.values())
        for key in self.quality_weights:
            self.quality_weights[key] /= quality_total
            
        # Initialisations des scorers individuels
        self.fundamental_scorer = FundamentalScorer(output_dir)
        self.technical_scorer = TechnicalScorer(output_dir)
        
    def score_stocks(
        self,
        fundamental_data: Dict[str, pd.DataFrame],
        technical_data: Dict[str, pd.DataFrame],
        quality_data: Optional[Dict[str, Dict[str, float]]] = None,
        market_data: Optional[pd.DataFrame] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Calcule les scores multifactoriels pour un ensemble d'actions.
        
        Args:
            fundamental_data: Dictionnaire de DataFrames avec les données fondamentales par ticker
            technical_data: Dictionnaire de DataFrames avec les données techniques par ticker
            quality_data: Dictionnaire des scores qualitatifs par ticker (optionnel)
            market_data: DataFrame avec les données de l'indice de référence (pour la force relative)
            save: Si True, sauvegarde les scores
            
        Returns:
            DataFrame avec les scores par action
        """
        # Calculer les scores fondamentaux
        fundamental_scores = self.fundamental_scorer.score_stocks(fundamental_data, save=False)
        
        # Calculer les scores techniques
        technical_scores = self.technical_scorer.score_stocks(technical_data, market_data, save=False)
        
        # Fusionner les scores
        all_scores = []
        
        # Liste de tous les tickers uniques
        all_tickers = list(set(
            list(fundamental_data.keys()) + 
            list(technical_data.keys()) +
            (list(quality_data.keys()) if quality_data else [])
        ))
        
        for ticker in all_tickers:
            try:
                # Initialiser les scores par défaut à None
                f_score = None
                t_score = None
                q_score = None
                
                # Récupérer le score fondamental si disponible
                if not fundamental_scores.empty and ticker in fundamental_scores['ticker'].values:
                    f_score = fundamental_scores.loc[fundamental_scores['ticker'] == ticker, 'overall_score'].iloc[0]
                
                # Récupérer le score technique si disponible
                if not technical_scores.empty and ticker in technical_scores['ticker'].values:
                    t_score = technical_scores.loc[technical_scores['ticker'] == ticker, 'overall_score'].iloc[0]
                
                # Récupérer ou calculer le score qualitatif si disponible
                if quality_data and ticker in quality_data:
                    q_score = self._calculate_quality_score(quality_data[ticker])
                
                # Calculer le score global multifactoriel
                overall_score = self._calculate_multifactor_score(f_score, t_score, q_score)
                
                # Ajouter à la liste des scores
                score_entry = {
                    'ticker': ticker,
                    'overall_score': overall_score,
                    'fundamental_score': f_score if f_score is not None else np.nan,
                    'technical_score': t_score if t_score is not None else np.nan,
                    'quality_score': q_score if q_score is not None else np.nan
                }
                
                all_scores.append(score_entry)
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul des scores multifactoriels pour {ticker}: {str(e)}")
                
        # Créer un DataFrame avec tous les scores
        if not all_scores:
            logger.warning("Aucun score multifactoriel n'a pu être calculé")
            return pd.DataFrame()
            
        scores_df = pd.DataFrame(all_scores)
        
        # Trier par score global décroissant
        scores_df = scores_df.sort_values('overall_score', ascending=False)
        
        # Sauvegarder les scores
        if save:
            date_str = pd.Timestamp.now().strftime('%Y%m%d')
            self.save_scores(scores_df, f"multifactor_scores_{date_str}.csv")
            
        return scores_df
        
    def _calculate_quality_score(self, quality_data: Dict[str, float]) -> float:
        """
        Calcule un score qualitatif pondéré.
        
        Args:
            quality_data: Dictionnaire des scores qualitatifs
            
        Returns:
            Score qualitatif pondéré entre 0 et 1
        """
        weighted_sum = 0
        total_applied_weight = 0
        
        for criterion, score in quality_data.items():
            if criterion in self.quality_weights:
                weighted_sum += score * self.quality_weights[criterion]
                total_applied_weight += self.quality_weights[criterion]
                
        # Si aucun poids n'a été appliqué, retourner 0.5 (valeur neutre)
        if total_applied_weight == 0:
            return 0.5
            
        # Normaliser par le poids total appliqué
        return weighted_sum / total_applied_weight
        
    def _calculate_multifactor_score(
        self,
        fundamental_score: Optional[float],
        technical_score: Optional[float],
        quality_score: Optional[float]
    ) -> float:
        """
        Calcule un score global multifactoriel pondéré.
        
        Args:
            fundamental_score: Score fondamental (optionnel)
            technical_score: Score technique (optionnel)
            quality_score: Score qualitatif (optionnel)
            
        Returns:
            Score global pondéré entre 0 et 1
        """
        weighted_sum = 0
        total_applied_weight = 0
        
        # Ajouter le score fondamental s'il est disponible
        if fundamental_score is not None:
            weighted_sum += fundamental_score * self.factor_weights.get('fundamental', 0)
            total_applied_weight += self.factor_weights.get('fundamental', 0)
            
        # Ajouter le score technique s'il est disponible
        if technical_score is not None:
            weighted_sum += technical_score * self.factor_weights.get('technical', 0)
            total_applied_weight += self.factor_weights.get('technical', 0)
            
        # Ajouter le score qualitatif s'il est disponible
        if quality_score is not None:
            weighted_sum += quality_score * self.factor_weights.get('quality', 0)
            total_applied_weight += self.factor_weights.get('quality', 0)
            
        # Si aucun poids n'a été appliqué, retourner 0.5 (valeur neutre)
        if total_applied_weight == 0:
            return 0.5
            
        # Normaliser par le poids total appliqué
        return weighted_sum / total_applied_weight