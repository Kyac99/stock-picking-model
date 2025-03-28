"""
Module pour la collecte de données financières à partir de différentes sources.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from alphavantage import AlphaVantage

# Configuration du logging
logger = logging.getLogger(__name__)

class DataCollector:
    """Classe de base pour tous les collecteurs de données."""
    
    def __init__(self, output_dir: str = "../data/raw"):
        """
        Initialise un collecteur de données.
        
        Args:
            output_dir: Répertoire où sauvegarder les données brutes
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_data(self, data: pd.DataFrame, filename: str) -> str:
        """
        Sauvegarde les données collectées dans un fichier.
        
        Args:
            data: DataFrame contenant les données à sauvegarder
            filename: Nom du fichier de sortie
            
        Returns:
            Chemin complet du fichier sauvegardé
        """
        filepath = os.path.join(self.output_dir, filename)
        data.to_csv(filepath, index=True)
        logger.info(f"Données sauvegardées dans {filepath}")
        return filepath


class YahooFinanceCollector(DataCollector):
    """Collecteur de données depuis Yahoo Finance via yfinance."""
    
    def __init__(self, output_dir: str = "../data/raw"):
        """Initialise le collecteur Yahoo Finance."""
        super().__init__(output_dir)
        # Utiliser yfinance avec pandas_datareader
        yf.pdr_override()
        
    def get_stock_data(
        self, 
        tickers: List[str], 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d",
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Récupère les données historiques pour une liste de tickers.
        
        Args:
            tickers: Liste des symboles d'actions à récupérer
            start_date: Date de début (par défaut: 1 an avant aujourd'hui)
            end_date: Date de fin (par défaut: aujourd'hui)
            interval: Intervalle des données ('1d', '1wk', '1mo', etc.)
            save: Si True, sauvegarde les données dans des fichiers CSV
            
        Returns:
            Dictionnaire de DataFrames avec les données par ticker
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
            
        if end_date is None:
            end_date = datetime.now()
            
        results = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Récupération des données pour {ticker}...")
                data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)
                
                if data.empty:
                    logger.warning(f"Aucune donnée récupérée pour {ticker}")
                    continue
                    
                results[ticker] = data
                
                if save:
                    filename = f"{ticker}_{interval}_{pd.to_datetime(start_date).strftime('%Y%m%d')}_to_{pd.to_datetime(end_date).strftime('%Y%m%d')}.csv"
                    self.save_data(data, filename)
                    
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des données pour {ticker}: {str(e)}")
                
        return results
        
    def get_fundamentals(self, tickers: List[str], save: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Récupère les données fondamentales pour une liste de tickers.
        
        Args:
            tickers: Liste des symboles d'actions à récupérer
            save: Si True, sauvegarde les données dans des fichiers JSON
            
        Returns:
            Dictionnaire de données fondamentales par ticker
        """
        fundamentals = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Récupération des fondamentaux pour {ticker}...")
                stock = yf.Ticker(ticker)
                
                # Récupérer les informations clés
                info = stock.info
                
                # Récupérer les états financiers
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow
                
                # Stocker toutes les données
                fundamentals[ticker] = {
                    'info': info,
                    'income_statement': income_stmt.to_dict() if not income_stmt.empty else {},
                    'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                    'cash_flow': cash_flow.to_dict() if not cash_flow.empty else {}
                }
                
                if save:
                    # Convertir chaque élément en DataFrame pour la sauvegarde
                    info_df = pd.DataFrame.from_dict(info, orient='index').T
                    self.save_data(info_df, f"{ticker}_info.csv")
                    
                    if not income_stmt.empty:
                        self.save_data(income_stmt, f"{ticker}_income_stmt.csv")
                    if not balance_sheet.empty:
                        self.save_data(balance_sheet, f"{ticker}_balance_sheet.csv")
                    if not cash_flow.empty:
                        self.save_data(cash_flow, f"{ticker}_cash_flow.csv")
                    
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des fondamentaux pour {ticker}: {str(e)}")
                
        return fundamentals


class AlphaVantageCollector(DataCollector):
    """Collecteur de données depuis Alpha Vantage."""
    
    def __init__(self, api_key: str, output_dir: str = "../data/raw"):
        """
        Initialise le collecteur Alpha Vantage.
        
        Args:
            api_key: Clé API Alpha Vantage
            output_dir: Répertoire où sauvegarder les données brutes
        """
        super().__init__(output_dir)
        self.av = AlphaVantage(api_key=api_key)
        
    def get_time_series(
        self, 
        symbol: str, 
        function: str = "TIME_SERIES_DAILY_ADJUSTED",
        outputsize: str = "full",
        save: bool = True
    ) -> pd.DataFrame:
        """
        Récupère les séries temporelles pour un symbole.
        
        Args:
            symbol: Symbole de l'action
            function: Type de série temporelle à récupérer
            outputsize: Taille de sortie ('compact' ou 'full')
            save: Si True, sauvegarde les données dans un fichier CSV
            
        Returns:
            DataFrame contenant les données de série temporelle
        """
        try:
            logger.info(f"Récupération des données {function} pour {symbol}...")
            data = self.av.data(function=function, symbol=symbol, outputsize=outputsize)
            
            if save and data is not None:
                filename = f"{symbol}_{function.lower()}.csv"
                self.save_data(data, filename)
                
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données {function} pour {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def get_fundamental_data(
        self,
        symbol: str,
        function: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Récupère les données fondamentales pour un symbole.
        
        Args:
            symbol: Symbole de l'action
            function: Type de données fondamentales à récupérer
            save: Si True, sauvegarde les données dans un fichier CSV
            
        Returns:
            DataFrame contenant les données fondamentales
        """
        try:
            logger.info(f"Récupération des données {function} pour {symbol}...")
            data = self.av.fundamentals(function=function, symbol=symbol)
            
            if save and data is not None:
                filename = f"{symbol}_{function.lower()}.csv"
                self.save_data(data, filename)
                
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données {function} pour {symbol}: {str(e)}")
            return pd.DataFrame()
