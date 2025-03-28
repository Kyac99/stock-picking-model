"""
Module pour le prétraitement des données financières brutes.
"""
import os
import logging
from typing import List, Dict, Optional, Union, Any

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

# Configuration du logging
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Classe de base pour les préprocesseurs de données."""
    
    def __init__(self, input_dir: str = "../data/raw", output_dir: str = "../data/processed"):
        """
        Initialise un préprocesseur de données.
        
        Args:
            input_dir: Répertoire contenant les données brutes
            output_dir: Répertoire où sauvegarder les données prétraitées
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_data(self, data: pd.DataFrame, filename: str) -> str:
        """
        Sauvegarde les données prétraitées dans un fichier.
        
        Args:
            data: DataFrame contenant les données à sauvegarder
            filename: Nom du fichier de sortie
            
        Returns:
            Chemin complet du fichier sauvegardé
        """
        filepath = os.path.join(self.output_dir, filename)
        data.to_csv(filepath, index=True)
        logger.info(f"Données prétraitées sauvegardées dans {filepath}")
        return filepath
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Charge des données à partir d'un fichier CSV.
        
        Args:
            filename: Nom du fichier à charger
            
        Returns:
            DataFrame contenant les données chargées
        """
        filepath = os.path.join(self.input_dir, filename)
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Données chargées depuis {filepath}")
            return data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données depuis {filepath}: {str(e)}")
            return pd.DataFrame()


class PriceDataPreprocessor(DataPreprocessor):
    """Préprocesseur pour les données de prix d'actions."""
    
    def __init__(
        self, 
        input_dir: str = "../data/raw", 
        output_dir: str = "../data/processed",
        handle_missing: bool = True,
        handle_outliers: bool = True
    ):
        """
        Initialise un préprocesseur de données de prix.
        
        Args:
            input_dir: Répertoire contenant les données brutes
            output_dir: Répertoire où sauvegarder les données prétraitées
            handle_missing: Si True, gère les valeurs manquantes
            handle_outliers: Si True, gère les valeurs aberrantes
        """
        super().__init__(input_dir, output_dir)
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        
    def preprocess(
        self, 
        filename: str, 
        save: bool = True,
        calculate_returns: bool = True,
        calculate_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Prétraite les données de prix d'actions.
        
        Args:
            filename: Nom du fichier contenant les données brutes
            save: Si True, sauvegarde les données prétraitées
            calculate_returns: Si True, calcule les rendements
            calculate_indicators: Si True, calcule les indicateurs techniques
            
        Returns:
            DataFrame contenant les données prétraitées
        """
        # Charger les données
        data = self.load_data(filename)
        if data.empty:
            return data
            
        # Gérer les valeurs manquantes
        if self.handle_missing:
            data = self._handle_missing_values(data)
            
        # Gérer les valeurs aberrantes
        if self.handle_outliers:
            data = self._handle_outliers(data)
            
        # Calculer les rendements
        if calculate_returns:
            data = self._calculate_returns(data)
            
        # Calculer les indicateurs techniques
        if calculate_indicators:
            data = self._calculate_technical_indicators(data)
            
        # Sauvegarder les données prétraitées
        if save:
            output_filename = f"processed_{filename}"
            self.save_data(data, output_filename)
            
        return data
        
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gère les valeurs manquantes dans les données.
        
        Args:
            data: DataFrame contenant les données
            
        Returns:
            DataFrame avec les valeurs manquantes traitées
        """
        # Vérifier s'il y a des valeurs manquantes
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Traitement de {missing_count} valeurs manquantes")
            
            # Interpolation pour les séries temporelles
            data = data.interpolate(method='time')
            
            # S'il reste des valeurs manquantes (début ou fin de série)
            if data.isnull().sum().sum() > 0:
                data = data.fillna(method='ffill').fillna(method='bfill')
                
        return data
        
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gère les valeurs aberrantes dans les données.
        
        Args:
            data: DataFrame contenant les données
            
        Returns:
            DataFrame avec les valeurs aberrantes traitées
        """
        # Méthode simple basée sur l'IQR pour les colonnes numériques
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Identifier les outliers
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"Détecté {outlier_count} valeurs aberrantes dans la colonne {col}")
                
                # Remplacer les outliers par les bornes
                data.loc[data[col] < lower_bound, col] = lower_bound
                data.loc[data[col] > upper_bound, col] = upper_bound
                
        return data
        
    def _calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule différentes mesures de rendement.
        
        Args:
            data: DataFrame contenant les données de prix
            
        Returns:
            DataFrame avec les rendements calculés
        """
        # Vérifier si les colonnes nécessaires existent
        if 'Close' not in data.columns and 'Adj Close' not in data.columns:
            logger.warning("Les colonnes 'Close' ou 'Adj Close' sont nécessaires pour calculer les rendements")
            return data
            
        # Utiliser 'Adj Close' si disponible, sinon 'Close'
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        
        # Rendements journaliers (pourcentage)
        data['daily_return'] = data[price_col].pct_change() * 100
        
        # Rendements sur plusieurs périodes
        data['weekly_return'] = data[price_col].pct_change(5) * 100
        data['monthly_return'] = data[price_col].pct_change(21) * 100
        data['quarterly_return'] = data[price_col].pct_change(63) * 100
        
        # Rendements logarithmiques
        data['log_return'] = np.log(data[price_col] / data[price_col].shift(1))
        
        # Rendements cumulés
        data['cumulative_return'] = (1 + data['daily_return'] / 100).cumprod() - 1
        
        return data
        
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques courants.
        
        Args:
            data: DataFrame contenant les données de prix
            
        Returns:
            DataFrame avec les indicateurs techniques
        """
        # Vérifier que les colonnes nécessaires existent
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Colonnes manquantes pour calculer les indicateurs techniques: {missing_cols}")
            return data
            
        # Copier le DataFrame pour éviter les problèmes de référence
        df = data.copy()
        
        # --- Moyennes mobiles ---
        # Simple Moving Average (SMA)
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Average (EMA)
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # --- Indicateurs de volatilité ---
        # Bollinger Bands
        df['BB_middle'] = df['SMA_20']
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        
        # Average True Range (ATR)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR_14'] = df['TR'].rolling(window=14).mean()
        
        # --- Indicateurs de tendance ---
        # MACD
        df['MACD_line'] = df['EMA_12'] - df['EMA_26'] if 'EMA_12' in df.columns and 'EMA_26' in df.columns else df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD_line'] - df['MACD_signal']
        
        # --- Indicateurs de momentum ---
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        df['lowest_14'] = df['Low'].rolling(window=14).min()
        df['highest_14'] = df['High'].rolling(window=14).max()
        df['%K'] = ((df['Close'] - df['lowest_14']) / (df['highest_14'] - df['lowest_14'])) * 100
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # --- Indicateurs de volume ---
        # On-Balance Volume (OBV)
        df['OBV'] = np.where(
            df['Close'] > df['Close'].shift(1),
            df['Volume'],
            np.where(
                df['Close'] < df['Close'].shift(1),
                -df['Volume'],
                0
            )
        ).cumsum()
        
        # Volume Moving Average
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        
        return df


class FundamentalDataPreprocessor(DataPreprocessor):
    """Préprocesseur pour les données fondamentales."""
    
    def __init__(
        self,
        input_dir: str = "../data/raw",
        output_dir: str = "../data/processed",
        scale_features: bool = True
    ):
        """
        Initialise un préprocesseur de données fondamentales.
        
        Args:
            input_dir: Répertoire contenant les données brutes
            output_dir: Répertoire où sauvegarder les données prétraitées
            scale_features: Si True, normalise les features
        """
        super().__init__(input_dir, output_dir)
        self.scale_features = scale_features
        self.scaler = RobustScaler()  # Plus robuste aux outliers que StandardScaler
        
    def preprocess_financials(
        self,
        income_filename: str,
        balance_filename: str,
        cashflow_filename: Optional[str] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Prétraite et combine les données financières fondamentales.
        
        Args:
            income_filename: Nom du fichier du compte de résultat
            balance_filename: Nom du fichier du bilan
            cashflow_filename: Nom du fichier du tableau de flux de trésorerie (optionnel)
            save: Si True, sauvegarde les données prétraitées
            
        Returns:
            DataFrame contenant les données financières combinées et prétraitées
        """
        # Charger les données
        income_data = self.load_data(income_filename)
        balance_data = self.load_data(balance_filename)
        
        # Vérifier si les données sont vides
        if income_data.empty or balance_data.empty:
            logger.error("Données de compte de résultat ou de bilan manquantes")
            return pd.DataFrame()
            
        # Charger les données de flux de trésorerie si disponibles
        cashflow_data = None
        if cashflow_filename:
            cashflow_data = self.load_data(cashflow_filename)
            
        # Calculer les ratios financiers
        financial_ratios = self._calculate_financial_ratios(income_data, balance_data, cashflow_data)
        
        # Normaliser les features si nécessaire
        if self.scale_features and not financial_ratios.empty:
            numeric_cols = financial_ratios.select_dtypes(include=[np.number]).columns
            financial_ratios[numeric_cols] = self.scaler.fit_transform(financial_ratios[numeric_cols])
            
        # Sauvegarder les données prétraitées
        if save and not financial_ratios.empty:
            ticker = income_filename.split('_')[0]  # Extraire le ticker du nom de fichier
            output_filename = f"{ticker}_financial_ratios.csv"
            self.save_data(financial_ratios, output_filename)
            
        return financial_ratios
        
    def _calculate_financial_ratios(
        self,
        income_data: pd.DataFrame,
        balance_data: pd.DataFrame,
        cashflow_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calcule les ratios financiers fondamentaux.
        
        Args:
            income_data: DataFrame du compte de résultat
            balance_data: DataFrame du bilan
            cashflow_data: DataFrame du tableau de flux de trésorerie (optionnel)
            
        Returns:
            DataFrame contenant les ratios financiers calculés
        """
        # Initialiser un DataFrame pour stocker les ratios
        ratios = pd.DataFrame(index=income_data.index)
        
        try:
            # --- Ratios de valorisation ---
            # P/E Ratio (Price-to-Earnings) - nécessite le prix de l'action
            if 'Net Income' in income_data.columns and 'Shares Outstanding' in income_data.columns:
                ratios['EPS'] = income_data['Net Income'] / income_data['Shares Outstanding']
            
            # Price-to-Book (P/B) - nécessite le prix de l'action
            if 'Total Stockholder Equity' in balance_data.columns and 'Shares Outstanding' in income_data.columns:
                ratios['BVPS'] = balance_data['Total Stockholder Equity'] / income_data['Shares Outstanding']
            
            # --- Ratios de rentabilité ---
            # Return on Equity (ROE)
            if 'Net Income' in income_data.columns and 'Total Stockholder Equity' in balance_data.columns:
                ratios['ROE'] = income_data['Net Income'] / balance_data['Total Stockholder Equity']
            
            # Return on Assets (ROA)
            if 'Net Income' in income_data.columns and 'Total Assets' in balance_data.columns:
                ratios['ROA'] = income_data['Net Income'] / balance_data['Total Assets']
            
            # Return on Invested Capital (ROIC)
            if ('Net Income' in income_data.columns and 
                'Total Debt' in balance_data.columns and 
                'Total Stockholder Equity' in balance_data.columns):
                ratios['ROIC'] = income_data['Net Income'] / (balance_data['Total Debt'] + balance_data['Total Stockholder Equity'])
            
            # Profit Margin
            if 'Net Income' in income_data.columns and 'Total Revenue' in income_data.columns:
                ratios['Profit_Margin'] = income_data['Net Income'] / income_data['Total Revenue']
            
            # Gross Margin
            if 'Gross Profit' in income_data.columns and 'Total Revenue' in income_data.columns:
                ratios['Gross_Margin'] = income_data['Gross Profit'] / income_data['Total Revenue']
            
            # Operating Margin
            if 'Operating Income' in income_data.columns and 'Total Revenue' in income_data.columns:
                ratios['Operating_Margin'] = income_data['Operating Income'] / income_data['Total Revenue']
            
            # --- Ratios de liquidité ---
            # Current Ratio
            if 'Current Assets' in balance_data.columns and 'Current Liabilities' in balance_data.columns:
                ratios['Current_Ratio'] = balance_data['Current Assets'] / balance_data['Current Liabilities']
            
            # Quick Ratio
            if ('Current Assets' in balance_data.columns and 
                'Inventory' in balance_data.columns and 
                'Current Liabilities' in balance_data.columns):
                ratios['Quick_Ratio'] = (balance_data['Current Assets'] - balance_data['Inventory']) / balance_data['Current Liabilities']
            
            # --- Ratios d'endettement ---
            # Debt-to-Equity (D/E)
            if 'Total Debt' in balance_data.columns and 'Total Stockholder Equity' in balance_data.columns:
                ratios['Debt_to_Equity'] = balance_data['Total Debt'] / balance_data['Total Stockholder Equity']
            
            # Debt-to-Assets
            if 'Total Debt' in balance_data.columns and 'Total Assets' in balance_data.columns:
                ratios['Debt_to_Assets'] = balance_data['Total Debt'] / balance_data['Total Assets']
            
            # Interest Coverage Ratio
            if 'Operating Income' in income_data.columns and 'Interest Expense' in income_data.columns:
                ratios['Interest_Coverage'] = income_data['Operating Income'] / income_data['Interest Expense']
            
            # --- Ratios d'efficacité ---
            # Asset Turnover
            if 'Total Revenue' in income_data.columns and 'Total Assets' in balance_data.columns:
                ratios['Asset_Turnover'] = income_data['Total Revenue'] / balance_data['Total Assets']
            
            # Inventory Turnover
            if 'Cost of Revenue' in income_data.columns and 'Inventory' in balance_data.columns:
                ratios['Inventory_Turnover'] = income_data['Cost of Revenue'] / balance_data['Inventory']
            
            # --- Ratios de croissance ---
            # Revenue Growth (YoY)
            if 'Total Revenue' in income_data.columns:
                ratios['Revenue_Growth'] = income_data['Total Revenue'].pct_change()
            
            # Earnings Growth (YoY)
            if 'Net Income' in income_data.columns:
                ratios['Earnings_Growth'] = income_data['Net Income'].pct_change()
            
            # --- Ratios de cash flow ---
            if cashflow_data is not None and not cashflow_data.empty:
                # Free Cash Flow (FCF)
                if 'Operating Cash Flow' in cashflow_data.columns and 'Capital Expenditure' in cashflow_data.columns:
                    ratios['FCF'] = cashflow_data['Operating Cash Flow'] - cashflow_data['Capital Expenditure']
                
                # FCF-to-Sales
                if 'FCF' in ratios.columns and 'Total Revenue' in income_data.columns:
                    ratios['FCF_to_Sales'] = ratios['FCF'] / income_data['Total Revenue']
                
                # FCF Yield - nécessite le prix de l'action
                if 'FCF' in ratios.columns and 'Shares Outstanding' in income_data.columns:
                    ratios['FCF_per_Share'] = ratios['FCF'] / income_data['Shares Outstanding']
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des ratios financiers: {str(e)}")
            
        return ratios
