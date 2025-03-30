"""
Tableau de bord interactif pour visualiser les résultats du modèle de stock picking.
Utilise Streamlit pour créer une interface utilisateur web.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.scoring import FundamentalScorer, TechnicalScorer, MultifactorScorer
from data.collectors import YahooFinanceCollector

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Stock Picking Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fonction pour charger les données de scores
@st.cache_data(ttl=3600)  # Cache valide pendant 1 heure
def load_scores(scores_path):
    """Charge les données de scores depuis un fichier CSV."""
    try:
        return pd.read_csv(scores_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement des scores: {e}")
        return pd.DataFrame()

# Fonction pour obtenir les données historiques d'une action
@st.cache_data(ttl=3600)  # Cache valide pendant 1 heure
def get_stock_data(ticker, period="1y"):
    """Récupère les données historiques d'une action depuis Yahoo Finance."""
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour {ticker}: {e}")
        return pd.DataFrame()

# Fonction pour créer un graphique interactif des scores
def plot_scores(scores_df, n_stocks=20, score_col='overall_score'):
    """Crée un graphique à barres des scores des meilleures actions."""
    # Prendre les n meilleures actions
    top_stocks = scores_df.sort_values(score_col, ascending=False).head(n_stocks)
    
    # Créer le graphique
    fig = px.bar(
        top_stocks,
        x='ticker',
        y=score_col,
        color=score_col,
        color_continuous_scale='RdYlGn',
        title=f"Top {n_stocks} Actions par Score"
    )
    
    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Score",
        coloraxis_colorbar=dict(title="Score"),
        height=500
    )
    
    return fig

# Fonction pour créer un graphique des prix historiques
def plot_price_history(data, ticker):
    """Crée un graphique des prix historiques d'une action."""
    if data.empty:
        return None
        
    fig = go.Figure()
    
    # Ajouter la trace du prix
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Prix de clôture',
            line=dict(color='royalblue', width=2)
        )
    )
    
    # Ajouter les moyennes mobiles si disponibles
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1.5)
            )
        )
    
    if 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='green', width=1.5)
            )
        )
    
    if 'SMA_200' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='red', width=1.5)
            )
        )
    
    # Mise en page
    fig.update_layout(
        title=f"Historique des prix pour {ticker}",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        height=500,
        hovermode="x unified"
    )
    
    return fig

# Fonction pour créer un graphique radar des sous-scores
def plot_radar_chart(scores_df, ticker):
    """Crée un graphique radar des sous-scores d'une action."""
    # Filtrer les données pour le ticker spécifié
    ticker_data = scores_df[scores_df['ticker'] == ticker]
    
    if ticker_data.empty:
        return None
        
    # Sélectionner les colonnes de scores (pas ticker ni overall_score)
    score_cols = [col for col in ticker_data.columns if col not in ['ticker', 'overall_score']]
    
    if not score_cols:
        return None
        
    values = ticker_data[score_cols].values.flatten().tolist()
    
    # Créer le graphique radar
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=score_cols,
        fill='toself',
        name=ticker
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f"Analyse détaillée des scores pour {ticker}",
        height=500
    )
    
    return fig

# Fonction pour créer une heatmap de corrélation entre les scores
def plot_correlation_heatmap(scores_df):
    """Crée une heatmap de corrélation entre les différents scores."""
    # Sélectionner les colonnes de scores (pas ticker)
    score_cols = [col for col in scores_df.columns if col != 'ticker']
    
    if len(score_cols) <= 1:
        return None
        
    # Calculer la matrice de corrélation
    corr_matrix = scores_df[score_cols].corr()
    
    # Créer la heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Corrélation entre les scores"
    )
    
    fig.update_layout(height=600)
    
    return fig

# Fonction principale pour l'application Streamlit
def main():
    """Fonction principale pour l'application Streamlit."""
    # Titre de l'application
    st.title("📊 Tableau de Bord du Modèle de Stock Picking")
    
    # Sidebar pour la navigation et les contrôles
    st.sidebar.title("Navigation")
    
    # Onglets principaux
    page = st.sidebar.radio(
        "Sélectionnez une page",
        ["Analyse des Scores", "Comparaison d'Actions", "Détails par Action", "Backtesting"]
    )
    
    # Charger les données de scores (simulées pour l'exemple)
    scores_path = st.sidebar.text_input(
        "Chemin vers le fichier de scores",
        "../data/results/multifactor_scores_20250330.csv"
    )
    
    # Bouton pour charger les données
    if st.sidebar.button("Charger les données"):
        scores_df = load_scores(scores_path)
        
        if not scores_df.empty:
            st.session_state['scores_df'] = scores_df
            st.sidebar.success(f"Données chargées: {len(scores_df)} actions trouvées.")
        else:
            st.sidebar.error("Aucune donnée n'a pu être chargée. Vérifiez le chemin du fichier.")
    
    # Afficher différentes pages en fonction de la sélection
    if page == "Analyse des Scores":
        show_scores_analysis()
    elif page == "Comparaison d'Actions":
        show_stocks_comparison()
    elif page == "Détails par Action":
        show_stock_details()
    elif page == "Backtesting":
        show_backtesting()

def show_scores_analysis():
    """Affiche la page d'analyse des scores."""
    st.header("Analyse des Scores")
    
    if 'scores_df' not in st.session_state:
        st.info("Veuillez charger les données de scores à partir du panneau latéral.")
        return
    
    scores_df = st.session_state['scores_df']
    
    # Options de filtrage
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_stocks = st.slider("Nombre d'actions à afficher", 5, 50, 20)
    
    with col2:
        score_col = st.selectbox(
            "Colonne de score à utiliser",
            ['overall_score'] + [col for col in scores_df.columns if col not in ['ticker', 'overall_score']]
        )
    
    with col3:
        min_score = st.slider("Score minimum", 0.0, 1.0, 0.0, 0.05)
        filtered_df = scores_df[scores_df[score_col] >= min_score]
    
    # Graphique des scores
    st.plotly_chart(plot_scores(filtered_df, n_stocks, score_col), use_container_width=True)
    
    # Statistiques des scores
    st.subheader("Statistiques des Scores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Score moyen", f"{scores_df[score_col].mean():.3f}")
        st.metric("Score médian", f"{scores_df[score_col].median():.3f}")
    
    with col2:
        st.metric("Score maximum", f"{scores_df[score_col].max():.3f}")
        st.metric("Score minimum", f"{scores_df[score_col].min():.3f}")
    
    # Heatmap de corrélation
    st.subheader("Corrélation entre les Scores")
    corr_fig = plot_correlation_heatmap(scores_df)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Pas assez de colonnes de scores pour calculer la corrélation.")
    
    # Tableau des scores
    st.subheader("Tableau des Scores")
    st.dataframe(filtered_df.sort_values(score_col, ascending=False))

def show_stocks_comparison():
    """Affiche la page de comparaison d'actions."""
    st.header("Comparaison d'Actions")
    
    if 'scores_df' not in st.session_state:
        st.info("Veuillez charger les données de scores à partir du panneau latéral.")
        return
    
    scores_df = st.session_state['scores_df']
    
    # Sélection des actions à comparer
    col1, col2 = st.columns(2)
    
    with col1:
        tickers = st.multiselect(
            "Sélectionnez les actions à comparer",
            options=scores_df['ticker'].unique(),
            default=scores_df.sort_values('overall_score', ascending=False).head(3)['ticker'].tolist()
        )
    
    with col2:
        period = st.selectbox(
            "Période historique",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3
        )
    
    if not tickers:
        st.warning("Veuillez sélectionner au moins une action.")
        return
    
    # Récupérer les données historiques
    data = {}
    for ticker in tickers:
        data[ticker] = get_stock_data(ticker, period)
    
    # Graphique des prix normalisés
    st.subheader("Comparaison des Prix Normalisés")
    
    # Créer un DataFrame pour les prix normalisés
    normalized_prices = pd.DataFrame(index=data[tickers[0]].index)
    
    for ticker in tickers:
        if not data[ticker].empty:
            normalized_prices[ticker] = data[ticker]['Close'] / data[ticker]['Close'].iloc[0] * 100
    
    # Créer le graphique
    fig = px.line(
        normalized_prices,
        title="Évolution des Prix Normalisés (Base 100)",
        labels={"value": "Prix (Base 100)", "index": "Date"}
    )
    
    fig.update_layout(
        height=500,
        hovermode="x unified",
        legend_title="Actions"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau de comparaison des scores
    st.subheader("Comparaison des Scores")
    
    comparison_df = scores_df[scores_df['ticker'].isin(tickers)].copy()
    comparison_df.set_index('ticker', inplace=True)
    
    # Tableau pivot pour faciliter la comparaison
    pivot_df = comparison_df.transpose()
    
    # Appliquer une mise en forme conditionnelle
    st.dataframe(pivot_df.style.highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='#FFB6C1'))
    
    # Graphique radar pour comparer les scores
    st.subheader("Comparaison des Scores par Catégorie")
    
    # Sélectionner les colonnes de scores (pas ticker ni overall_score)
    score_cols = [col for col in scores_df.columns if col not in ['ticker', 'overall_score']]
    
    if score_cols:
        radar_fig = go.Figure()
        
        for ticker in tickers:
            ticker_data = scores_df[scores_df['ticker'] == ticker]
            if not ticker_data.empty:
                values = ticker_data[score_cols].values.flatten().tolist()
                radar_fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=score_cols,
                    fill='toself',
                    name=ticker
                ))
        
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Comparaison des Scores par Catégorie",
            height=600
        )
        
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Pas assez de colonnes de scores pour créer un graphique radar.")

def show_stock_details():
    """Affiche la page de détails par action."""
    st.header("Détails par Action")
    
    if 'scores_df' not in st.session_state:
        st.info("Veuillez charger les données de scores à partir du panneau latéral.")
        return
    
    scores_df = st.session_state['scores_df']
    
    # Sélection de l'action
    ticker = st.selectbox(
        "Sélectionnez une action",
        options=scores_df['ticker'].unique(),
        index=0
    )
    
    if not ticker:
        return
    
    # Récupérer les données de l'action
    col1, col2 = st.columns([1, 2])
    
    with col1:
        period = st.selectbox(
            "Période historique",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3
        )
        
        # Afficher les scores de l'action
        st.subheader("Scores")
        
        ticker_scores = scores_df[scores_df['ticker'] == ticker].iloc[0]
        
        # Score global
        st.metric(
            "Score Global",
            f"{ticker_scores['overall_score']:.3f}",
            delta=f"{ticker_scores['overall_score'] - scores_df['overall_score'].mean():.3f}",
            delta_color="normal"
        )
        
        # Autres scores
        score_cols = [col for col in scores_df.columns if col not in ['ticker', 'overall_score']]
        
        for col in score_cols:
            if col in ticker_scores:
                st.metric(
                    col,
                    f"{ticker_scores[col]:.3f}",
                    delta=f"{ticker_scores[col] - scores_df[col].mean():.3f}",
                    delta_color="normal"
                )
    
    with col2:
        # Récupérer et afficher les données historiques
        stock_data = get_stock_data(ticker, period)
        
        if not stock_data.empty:
            # Graphique des prix
            price_fig = plot_price_history(stock_data, ticker)
            st.plotly_chart(price_fig, use_container_width=True)
            
            # Indicateurs de performance
            if len(stock_data) > 1:
                # Calculer les rendements
                returns = stock_data['Close'].pct_change().dropna()
                
                # Métriques de performance
                st.subheader("Indicateurs de Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Rendement total
                    total_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
                    st.metric("Rendement Total", f"{total_return:.2f}%")
                
                with col2:
                    # Volatilité annualisée
                    volatility = returns.std() * np.sqrt(252) * 100
                    st.metric("Volatilité Annualisée", f"{volatility:.2f}%")
                
                with col3:
                    # Ratio de Sharpe (simplifié)
                    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                    st.metric("Ratio de Sharpe", f"{sharpe:.2f}")
                
                # Graphique de rendements
                returns_fig = px.histogram(
                    returns,
                    nbins=50,
                    title=f"Distribution des Rendements Journaliers pour {ticker}",
                    labels={"value": "Rendement (%)", "count": "Fréquence"}
                )
                
                returns_fig.update_layout(height=400)
                
                st.plotly_chart(returns_fig, use_container_width=True)
                
                # Graphique radar des scores
                radar_fig = plot_radar_chart(scores_df, ticker)
                
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.error(f"Impossible de récupérer les données pour {ticker}")

def show_backtesting():
    """Affiche la page de backtesting."""
    st.header("Backtesting de la Stratégie")
    
    if 'scores_df' not in st.session_state:
        st.info("Veuillez charger les données de scores à partir du panneau latéral.")
        return
    
    scores_df = st.session_state['scores_df']
    
    # Paramètres de backtesting
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_stocks = st.slider("Nombre d'actions dans le portefeuille", 5, 30, 10)
    
    with col2:
        lookback_period = st.selectbox(
            "Période de backtesting",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
    
    with col3:
        rebalance_freq = st.selectbox(
            "Fréquence de rééquilibrage",
            options=["Quotidienne", "Hebdomadaire", "Mensuelle", "Trimestrielle"],
            index=2
        )
    
    # Sélection des actions pour le backtesting
    top_stocks = scores_df.sort_values('overall_score', ascending=False).head(n_stocks)['ticker'].tolist()
    
    st.subheader(f"Top {n_stocks} Actions pour le Backtesting")
    st.write(", ".join(top_stocks))
    
    # Simuler le backtesting (simplifié)
    st.info("Simulation de backtesting en cours... Cette fonctionnalité sera implémentée ultérieurement.")
    
    # Placeholder pour les résultats de backtesting
    st.subheader("Résultats du Backtesting")
    
    # Créer un graphique exemple
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    
    # Simuler les performances du portefeuille et du benchmark
    np.random.seed(42)  # Pour la reproducibilité
    portfolio_returns = np.random.normal(0.0003, 0.01, len(dates))
    benchmark_returns = np.random.normal(0.0002, 0.01, len(dates))
    
    # Créer les indices cumulatifs
    portfolio_index = (1 + pd.Series(portfolio_returns)).cumprod() * 100
    benchmark_index = (1 + pd.Series(benchmark_returns)).cumprod() * 100
    
    # Créer le DataFrame
    backtesting_df = pd.DataFrame({
        'Date': dates,
        'Portfolio': portfolio_index,
        'Benchmark': benchmark_index
    })
    
    # Créer le graphique
    fig = px.line(
        backtesting_df,
        x='Date',
        y=['Portfolio', 'Benchmark'],
        title="Performance du Portefeuille vs. Benchmark (Simulation)",
        labels={"value": "Performance", "variable": ""}
    )
    
    fig.update_layout(height=500, hovermode="x unified")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques de performance
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Rendement Annualisé",
            f"{((portfolio_index.iloc[-1] / 100) ** (252 / len(dates)) - 1) * 100:.2f}%",
            f"{((portfolio_index.iloc[-1] / benchmark_index.iloc[-1]) - 1) * 100:.2f}%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Volatilité Annualisée",
            f"{np.std(portfolio_returns) * np.sqrt(252) * 100:.2f}%",
            f"{(np.std(portfolio_returns) - np.std(benchmark_returns)) * np.sqrt(252) * 100:.2f}%",
            delta_color="inverse"
        )
    
    with col3:
        portfolio_sharpe = (np.mean(portfolio_returns) * 252) / (np.std(portfolio_returns) * np.sqrt(252))
        benchmark_sharpe = (np.mean(benchmark_returns) * 252) / (np.std(benchmark_returns) * np.sqrt(252))
        
        st.metric(
            "Ratio de Sharpe",
            f"{portfolio_sharpe:.2f}",
            f"{portfolio_sharpe - benchmark_sharpe:.2f}",
            delta_color="normal"
        )
    
    with col4:
        # Maximum Drawdown
        def calculate_max_drawdown(returns):
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            return drawdown.min() * 100
        
        portfolio_mdd = calculate_max_drawdown(portfolio_returns)
        benchmark_mdd = calculate_max_drawdown(benchmark_returns)
        
        st.metric(
            "Drawdown Maximum",
            f"{portfolio_mdd:.2f}%",
            f"{portfolio_mdd - benchmark_mdd:.2f}%",
            delta_color="inverse"
        )
    
    # Tableau des positions
    st.subheader("Positions Actuelles (Simulation)")
    
    # Simuler les positions actuelles
    positions = []
    
    for ticker in top_stocks:
        # Simuler une allocation aléatoire
        allocation = np.random.uniform(0.05, 0.15)
        
        # Simuler un rendement
        performance = np.random.uniform(-0.1, 0.2)
        
        positions.append({
            'Ticker': ticker,
            'Allocation': allocation,
            'Performance': performance,
            'Score': scores_df[scores_df['ticker'] == ticker]['overall_score'].values[0]
        })
    
    # Normaliser les allocations pour qu'elles somment à 1
    total_allocation = sum(pos['Allocation'] for pos in positions)
    
    for pos in positions:
        pos['Allocation'] /= total_allocation
    
    # Créer le DataFrame
    positions_df = pd.DataFrame(positions)
    
    # Afficher le tableau
    st.dataframe(
        positions_df.style.format({
            'Allocation': '{:.2%}',
            'Performance': '{:.2%}',
            'Score': '{:.3f}'
        }).bar(subset=['Allocation'], color='#90ee90').bar(subset=['Performance'], color=['#ff9999', '#90ee90'])
    )

# Point d'entrée de l'application
if __name__ == "__main__":
    main()
