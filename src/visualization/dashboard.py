"""
Module pour créer un tableau de bord interactif pour visualiser les résultats du modèle de stock picking.
"""
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

class StockPickingDashboard:
    """Classe pour créer un tableau de bord interactif pour visualiser les résultats du stock picking."""
    
    def __init__(
        self,
        results_dir: str = "../data/results",
        asset_dir: str = "../data/assets"
    ):
        """
        Initialise le tableau de bord de stock picking.
        
        Args:
            results_dir: Répertoire contenant les résultats du modèle
            asset_dir: Répertoire pour les actifs statiques (images, etc.)
        """
        self.results_dir = results_dir
        self.asset_dir = asset_dir
        
        # Création de l'application Dash
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.FLATLY],
            suppress_callback_exceptions=True
        )
        
        self.app.title = "Stock Picking Dashboard"
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Configure la mise en page du tableau de bord."""
        self.app.layout = dbc.Container(
            [
                # En-tête
                dbc.Row(
                    dbc.Col(
                        html.H1("Stock Picking Dashboard", className="text-center my-4"),
                        width=12
                    )
                ),
                
                # Filtres
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Filtres", className="mb-3"),
                                
                                html.Label("Type de score:"),
                                dcc.Dropdown(
                                    id="score-type-dropdown",
                                    options=[
                                        {"label": "Multifactoriel", "value": "multifactor"},
                                        {"label": "Fondamental", "value": "fundamental"},
                                        {"label": "Technique", "value": "technical"}
                                    ],
                                    value="multifactor",
                                    className="mb-3"
                                ),
                                
                                html.Label("Score minimum:"),
                                dcc.Slider(
                                    id="min-score-slider",
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    value=0.5,
                                    marks={i/10: str(i/10) for i in range(0, 11, 2)},
                                    className="mb-3"
                                ),
                                
                                html.Label("Nombre de résultats:"),
                                dcc.Slider(
                                    id="num-results-slider",
                                    min=5,
                                    max=50,
                                    step=5,
                                    value=20,
                                    marks={i: str(i) for i in range(5, 51, 5)},
                                    className="mb-3"
                                )
                            ],
                            width=3
                        ),
                        
                        # Graphique des meilleurs scores
                        dbc.Col(
                            [
                                html.H5("Meilleurs scores", className="mb-3"),
                                dcc.Graph(id="top-scores-graph")
                            ],
                            width=9
                        )
                    ],
                    className="mb-4"
                ),
                
                # Tableau des résultats
                dbc.Row(
                    dbc.Col(
                        [
                            html.H5("Résultats détaillés", className="mb-3"),
                            html.Div(id="results-table")
                        ],
                        width=12
                    ),
                    className="mb-4"
                ),
                
                # Graphique radar pour la comparaison des actions
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Analyse comparative", className="mb-3"),
                                html.Label("Sélectionnez les actions à comparer:"),
                                dcc.Dropdown(
                                    id="stock-comparison-dropdown",
                                    multi=True,
                                    className="mb-3"
                                )
                            ],
                            width=3
                        ),
                        
                        dbc.Col(
                            dcc.Graph(id="comparison-radar-chart"),
                            width=9
                        )
                    ],
                    className="mb-4"
                ),
                
                # Distribution des scores
                dbc.Row(
                    dbc.Col(
                        [
                            html.H5("Distribution des scores", className="mb-3"),
                            dcc.Graph(id="score-distribution-histogram")
                        ],
                        width=12
                    ),
                    className="mb-4"
                ),
                
                # Pied de page
                dbc.Row(
                    dbc.Col(
                        html.P(
                            "© 2025 - Modèle de Stock Picking - Développé par l'équipe Quant Finance",
                            className="text-center text-muted"
                        ),
                        width=12
                    )
                )
            ],
            fluid=True,
            className="px-4 py-3"
        )
        
    def setup_callbacks(self):
        """Configure les callbacks pour l'interactivité du tableau de bord."""
        
        # Callback pour mettre à jour le graphique des meilleurs scores
        @self.app.callback(
            Output("top-scores-graph", "figure"),
            [
                Input("score-type-dropdown", "value"),
                Input("min-score-slider", "value"),
                Input("num-results-slider", "value")
            ]
        )
        def update_top_scores_graph(score_type, min_score, num_results):
            # Charger les données
            df = self.load_score_data(score_type)
            
            if df is None or df.empty:
                return self.empty_figure("Aucune donnée disponible")
                
            # Filtrer les données
            filtered_df = df[df["overall_score"] >= min_score].sort_values("overall_score", ascending=False).head(num_results)
            
            if filtered_df.empty:
                return self.empty_figure("Aucun résultat ne correspond aux critères de filtrage")
                
            # Créer le graphique
            fig = px.bar(
                filtered_df,
                x="ticker",
                y="overall_score",
                color="overall_score",
                color_continuous_scale="viridis",
                labels={"ticker": "Symbole", "overall_score": "Score global"},
                title=f"Top {num_results} actions par score {self.get_score_type_label(score_type)}"
            )
            
            fig.update_layout(
                coloraxis_showscale=True,
                xaxis_tickangle=-45,
                yaxis_range=[0, 1]
            )
            
            return fig
            
        # Callback pour mettre à jour le tableau des résultats
        @self.app.callback(
            Output("results-table", "children"),
            [
                Input("score-type-dropdown", "value"),
                Input("min-score-slider", "value"),
                Input("num-results-slider", "value")
            ]
        )
        def update_results_table(score_type, min_score, num_results):
            # Charger les données
            df = self.load_score_data(score_type)
            
            if df is None or df.empty:
                return html.Div("Aucune donnée disponible", className="text-center text-muted")
                
            # Filtrer les données
            filtered_df = df[df["overall_score"] >= min_score].sort_values("overall_score", ascending=False).head(num_results)
            
            if filtered_df.empty:
                return html.Div("Aucun résultat ne correspond aux critères de filtrage", className="text-center text-muted")
                
            # Formater le tableau
            columns_to_display = ["ticker", "overall_score"]
            if score_type == "multifactor":
                columns_to_display.extend(["fundamental_score", "technical_score", "quality_score"])
                
            formatted_df = filtered_df[columns_to_display].copy()
            
            # Renommer les colonnes
            formatted_df.columns = [self.get_column_label(col) for col in formatted_df.columns]
            
            # Formater les scores
            for col in formatted_df.columns:
                if "Score" in col:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                    
            # Créer le tableau HTML
            table = dbc.Table.from_dataframe(
                formatted_df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                className="text-center"
            )
            
            return table
            
        # Callback pour mettre à jour le dropdown de comparaison des actions
        @self.app.callback(
            Output("stock-comparison-dropdown", "options"),
            [Input("score-type-dropdown", "value")]
        )
        def update_comparison_dropdown(score_type):
            # Charger les données
            df = self.load_score_data(score_type)
            
            if df is None or df.empty:
                return []
                
            # Créer les options du dropdown
            options = [{"label": ticker, "value": ticker} for ticker in df["ticker"]]
            
            return options
            
        # Callback pour mettre à jour le graphique radar de comparaison
        @self.app.callback(
            Output("comparison-radar-chart", "figure"),
            [
                Input("stock-comparison-dropdown", "value"),
                Input("score-type-dropdown", "value")
            ]
        )
        def update_comparison_radar(selected_stocks, score_type):
            if not selected_stocks or len(selected_stocks) == 0:
                return self.empty_figure("Sélectionnez des actions à comparer")
                
            # Charger les données
            df = self.load_score_data(score_type)
            
            if df is None or df.empty:
                return self.empty_figure("Aucune donnée disponible")
                
            # Filtrer les données pour les actions sélectionnées
            filtered_df = df[df["ticker"].isin(selected_stocks)]
            
            if filtered_df.empty:
                return self.empty_figure("Aucune donnée disponible pour les actions sélectionnées")
                
            # Définir les catégories pour le graphique radar
            if score_type == "multifactor":
                categories = ["overall_score", "fundamental_score", "technical_score", "quality_score"]
            elif score_type == "fundamental":
                # Prendre les 6 premiers scores fondamentaux
                categories = ["overall_score"] + list(filtered_df.columns[2:8])
            else:  # score_type == "technical"
                # Prendre les 6 premiers scores techniques
                categories = ["overall_score"] + list(filtered_df.columns[2:8])
                
            # Créer le graphique radar
            fig = go.Figure()
            
            for _, row in filtered_df.iterrows():
                data = []
                for cat in categories:
                    if cat in row and not pd.isna(row[cat]):
                        data.append(row[cat])
                    else:
                        data.append(0)
                        
                # Ajouter une trace pour chaque action
                fig.add_trace(go.Scatterpolar(
                    r=data,
                    theta=[self.get_column_label(cat) for cat in categories],
                    fill='toself',
                    name=row["ticker"]
                ))
                
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title=f"Comparaison des scores {self.get_score_type_label(score_type)}"
            )
            
            return fig
            
        # Callback pour mettre à jour l'histogramme de distribution des scores
        @self.app.callback(
            Output("score-distribution-histogram", "figure"),
            [Input("score-type-dropdown", "value")]
        )
        def update_score_distribution(score_type):
            # Charger les données
            df = self.load_score_data(score_type)
            
            if df is None or df.empty:
                return self.empty_figure("Aucune donnée disponible")
                
            # Créer l'histogramme
            fig = px.histogram(
                df,
                x="overall_score",
                nbins=20,
                range_x=[0, 1],
                labels={"overall_score": "Score global"},
                title=f"Distribution des scores {self.get_score_type_label(score_type)}"
            )
            
            fig.update_layout(
                bargap=0.2,
                xaxis_range=[0, 1]
            )
            
            return fig
            
    def load_score_data(self, score_type: str) -> pd.DataFrame:
        """
        Charge les données de score depuis les fichiers CSV.
        
        Args:
            score_type: Type de score à charger ('multifactor', 'fundamental' ou 'technical')
            
        Returns:
            DataFrame contenant les données de score
        """
        # Obtenir le chemin du fichier le plus récent pour le type de score
        file_pattern = f"{score_type}_scores_*.csv"
        matching_files = []
        
        for filename in os.listdir(self.results_dir):
            if filename.startswith(f"{score_type}_scores_") and filename.endswith(".csv"):
                matching_files.append(os.path.join(self.results_dir, filename))
                
        if not matching_files:
            return None
            
        # Trier par date de modification (le plus récent en premier)
        most_recent_file = max(matching_files, key=os.path.getmtime)
        
        # Charger le fichier
        try:
            df = pd.read_csv(most_recent_file)
            return df
        except Exception as e:
            print(f"Erreur lors du chargement du fichier {most_recent_file}: {str(e)}")
            return None
            
    def get_score_type_label(self, score_type: str) -> str:
        """
        Retourne le libellé du type de score.
        
        Args:
            score_type: Type de score ('multifactor', 'fundamental' ou 'technical')
            
        Returns:
            Libellé du type de score
        """
        labels = {
            "multifactor": "multifactoriels",
            "fundamental": "fondamentaux",
            "technical": "techniques"
        }
        
        return labels.get(score_type, score_type)
        
    def get_column_label(self, column: str) -> str:
        """
        Retourne le libellé d'une colonne pour l'affichage.
        
        Args:
            column: Nom de la colonne
            
        Returns:
            Libellé de la colonne
        """
        labels = {
            "ticker": "Symbole",
            "overall_score": "Score global",
            "fundamental_score": "Score fondamental",
            "technical_score": "Score technique",
            "quality_score": "Score qualitatif",
            
            # Scores fondamentaux
            "PE_Ratio": "P/E Ratio",
            "PB_Ratio": "P/B Ratio",
            "EV_EBITDA": "EV/EBITDA",
            "Price_to_Sales": "Price/Sales",
            "ROE": "ROE",
            "ROA": "ROA",
            "ROIC": "ROIC",
            "Profit_Margin": "Marge bénéficiaire",
            "Revenue_Growth": "Croissance des revenus",
            "Earnings_Growth": "Croissance des résultats",
            "Debt_to_Equity": "Dette/Fonds propres",
            "Current_Ratio": "Ratio de liquidité",
            "Interest_Coverage": "Couverture des intérêts",
            
            # Scores techniques
            "trend_ma": "Tendance (MM)",
            "macd": "MACD",
            "rsi": "RSI",
            "stochastic": "Stochastique",
            "bollinger": "Bandes de Bollinger",
            "atr": "ATR",
            "obv": "OBV",
            "volume_ma": "Volume (MM)",
            "relative_strength": "Force relative"
        }
        
        return labels.get(column, column)
        
    def empty_figure(self, message: str) -> go.Figure:
        """
        Crée une figure vide avec un message.
        
        Args:
            message: Message à afficher
            
        Returns:
            Figure Plotly vide avec un message
        """
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16)
        )
        
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        return fig
        
    def run_server(self, debug: bool = True, port: int = 8050):
        """
        Lance le serveur du tableau de bord.
        
        Args:
            debug: Mode debug actif ou non
            port: Port sur lequel lancer le serveur
        """
        self.app.run_server(debug=debug, port=port)
        
        
# Point d'entrée pour lancer le tableau de bord
if __name__ == "__main__":
    dashboard = StockPickingDashboard()
    dashboard.run_server()
