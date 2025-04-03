                # Personnaliser le graphique
                fig.update_layout(
                    title="Décomposition des scores pour le Top 10",
                    xaxis_title="Contribution au score final",
                    yaxis_title="Ticker",
                    barmode='stack',
                    height=500,
                    yaxis={'categoryorder':'total ascending'}
                )
                
                # Afficher le graphique
                st.plotly_chart(fig, use_container_width=True)
                
                # Permettre de télécharger les scores
                st.download_button(
                    label="📥 Télécharger les scores",
                    data=sorted_scores.to_csv(index=True),
                    file_name="stock_scores.csv",
                    mime="text/csv"
                )
                
                # Analyse détaillée d'une action
                st.subheader("Analyse détaillée d'une action")
                
                # Sélectionner une action pour l'analyse détaillée
                selected_ticker = st.selectbox(
                    "Sélectionner une action à analyser en détail",
                    options=sorted_scores.index.tolist(),
                    index=0  # Par défaut, la première action
                )
                
                if selected_ticker:
                    # Créer deux colonnes pour l'affichage
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Afficher les scores pour l'action sélectionnée
                        st.subheader(f"Scores de {selected_ticker}")
                        
                        # Créer un DataFrame pour afficher les scores
                        scores_display = pd.DataFrame({
                            'Score': [
                                sorted_scores.loc[selected_ticker, 'Fundamental_Score'],
                                sorted_scores.loc[selected_ticker, 'Technical_Score'],
                                sorted_scores.loc[selected_ticker, 'Quality_Score'] if 'Quality_Score' in sorted_scores.columns else 0,
                                sorted_scores.loc[selected_ticker, 'Combined_Score']
                            ],
                            'Poids': [
                                fundamental_weight,
                                technical_weight,
                                quality_weight,
                                1.0
                            ],
                            'Contribution': [
                                sorted_scores.loc[selected_ticker, 'Fundamental_Score'] * fundamental_weight,
                                sorted_scores.loc[selected_ticker, 'Technical_Score'] * technical_weight,
                                sorted_scores.loc[selected_ticker, 'Quality_Score'] * quality_weight if 'Quality_Score' in sorted_scores.columns else 0,
                                sorted_scores.loc[selected_ticker, 'Combined_Score']
                            ]
                        }, index=['Fondamental', 'Technique', 'Qualitatif', 'Combiné'])
                        
                        # Afficher le tableau de scores
                        st.dataframe(scores_display.style.format({
                            'Score': '{:.4f}',
                            'Poids': '{:.2f}',
                            'Contribution': '{:.4f}'
                        }), use_container_width=True)
                        
                        # Afficher les données fondamentales de l'action
                        st.subheader(f"Fondamentaux de {selected_ticker}")
                        
                        if selected_ticker in st.session_state.clean_fund_data.index:
                            # Sélectionner les métriques importantes
                            key_metrics = ["PE_Ratio", "PB_Ratio", "ROE", "ROA", "ROIC", "Profit_Margin", 
                                          "Revenue_Growth", "Earnings_Growth", "Debt_to_Equity"]
                            
                            # Filtrer les métriques disponibles
                            available_metrics = [m for m in key_metrics if m in st.session_state.clean_fund_data.columns]
                            
                            if available_metrics:
                                # Créer un DataFrame pour l'affichage
                                fund_display = pd.DataFrame({
                                    'Valeur': [st.session_state.clean_fund_data.loc[selected_ticker, m] for m in available_metrics]
                                }, index=available_metrics)
                                
                                # Afficher le tableau
                                st.dataframe(fund_display, use_container_width=True)
                            else:
                                st.warning("Aucune métrique fondamentale disponible pour cette action.")
                        else:
                            st.warning(f"Aucune donnée fondamentale disponible pour {selected_ticker}.")
                    
                    with col2:
                        # Afficher le graphique de l'évolution du prix
                        st.subheader(f"Évolution du prix de {selected_ticker}")
                        
                        if selected_ticker in st.session_state.clean_price_data.columns:
                            # Obtenir les prix
                            prices = st.session_state.clean_price_data[selected_ticker]
                            
                            # Calculer les moyennes mobiles
                            ma_20 = prices.rolling(window=20).mean()
                            ma_50 = prices.rolling(window=50).mean()
                            ma_200 = prices.rolling(window=200).mean()
                            
                            # Créer le graphique avec Plotly
                            fig = go.Figure()
                            
                            # Ajouter le prix
                            fig.add_trace(go.Scatter(
                                x=prices.index,
                                y=prices.values,
                                name=f"{selected_ticker} Prix",
                                line=dict(color='royalblue')
                            ))
                            
                            # Ajouter les moyennes mobiles
                            fig.add_trace(go.Scatter(
                                x=ma_20.index,
                                y=ma_20.values,
                                name="MA 20 jours",
                                line=dict(color='orange', dash='dash')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=ma_50.index,
                                y=ma_50.values,
                                name="MA 50 jours",
                                line=dict(color='green', dash='dash')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=ma_200.index,
                                y=ma_200.values,
                                name="MA 200 jours",
                                line=dict(color='red', dash='dash')
                            ))
                            
                            # Personnaliser le graphique
                            fig.update_layout(
                                title=f"Évolution du prix de {selected_ticker} avec moyennes mobiles",
                                xaxis_title="Date",
                                yaxis_title="Prix",
                                height=400
                            )
                            
                            # Afficher le graphique
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Ajouter un graphique technique (RSI par exemple)
                            st.subheader(f"Analyse technique de {selected_ticker}")
                            
                            # Calculer le RSI (Relative Strength Index)
                            delta = prices.diff()
                            gain = delta.clip(lower=0)
                            loss = -delta.clip(upper=0)
                            avg_gain = gain.rolling(window=14).mean()
                            avg_loss = loss.rolling(window=14).mean()
                            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Éviter division par zéro
                            rsi = 100 - (100 / (1 + rs))
                            
                            # Créer le graphique RSI
                            fig = go.Figure()
                            
                            # Ajouter le RSI
                            fig.add_trace(go.Scatter(
                                x=rsi.index,
                                y=rsi.values,
                                name="RSI",
                                line=dict(color='purple')
                            ))
                            
                            # Ajouter des lignes horizontales pour les seuils
                            fig.add_shape(
                                type="line",
                                x0=rsi.index[0],
                                y0=70,
                                x1=rsi.index[-1],
                                y1=70,
                                line=dict(color="red", width=1, dash="dash")
                            )
                            
                            fig.add_shape(
                                type="line",
                                x0=rsi.index[0],
                                y0=30,
                                x1=rsi.index[-1],
                                y1=30,
                                line=dict(color="green", width=1, dash="dash")
                            )
                            
                            # Personnaliser le graphique
                            fig.update_layout(
                                title=f"Relative Strength Index (RSI) de {selected_ticker}",
                                xaxis_title="Date",
                                yaxis_title="RSI",
                                height=300,
                                yaxis=dict(range=[0, 100])
                            )
                            
                            # Afficher le graphique
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Aucune donnée de prix disponible pour {selected_ticker}.")
            else:
                st.error("Aucun score n'a pu être calculé. Vérifiez les données d'entrée.")

# Onglet 3: Optimisation de portefeuille
with tab3:
    st.header("🔮 Optimisation de Portefeuille")
    st.markdown("Cette section utilise les scores calculés pour optimiser l'allocation d'actifs selon différentes méthodes.")
    
    # Vérifier si le scoring a été effectué
    if 'scoring_done' not in st.session_state or not st.session_state.scoring_done:
        st.warning("⚠️ Veuillez d'abord effectuer le scoring des actions dans l'onglet 'Analyse & Scoring'.")
    else:
        # Paramètres d'optimisation
        st.subheader("Paramètres d'optimisation")
        
        # Sélectionner la méthode d'optimisation
        optimization_method = st.radio(
            "Méthode d'optimisation",
            options=["Markowitz", "Score-Based", "Risk-Parity"],
            index=1,  # Par défaut: Score-Based
            horizontal=True
        )
        
        # Description de la méthode sélectionnée
        method_descriptions = {
            "Markowitz": "Optimisation basée sur la théorie moderne du portefeuille, maximisant le ratio de Sharpe (rendement ajusté au risque).",
            "Score-Based": "Optimisation tenant compte des scores calculés, favorisant les actions avec les meilleurs scores tout en minimisant le risque.",
            "Risk-Parity": "Allocation égale du risque entre les actifs, particulièrement utile pour diversifier le risque."
        }
        
        st.info(method_descriptions[optimization_method])
        
        # Sélectionner le nombre d'actions à inclure
        top_n = st.slider(
            "Nombre d'actions à inclure dans le portefeuille",
            min_value=5,
            max_value=min(20, len(st.session_state.combined_scores)),
            value=10,
            step=1
        )
        
        # Bouton pour lancer l'optimisation
        if st.button("🚀 Lancer l'optimisation du portefeuille"):
            with st.spinner("Optimisation en cours..."):
                try:
                    # Sélectionner les top N actions selon le score combiné
                    top_tickers = st.session_state.combined_scores.sort_values(
                        by="Combined_Score", ascending=False
                    ).index[:top_n]
                    
                    # Filtrer les données de prix pour les tickers sélectionnés
                    top_price_data = st.session_state.clean_price_data[top_tickers]
                    
                    # Extraire les scores combinés pour les top tickers
                    top_scores = st.session_state.combined_scores.loc[top_tickers, "Combined_Score"]
                    
                    # Optimiser le portefeuille selon la méthode sélectionnée
                    weights, exp_return, exp_risk, sharpe, details = optimize_portfolio(
                        price_data=top_price_data,
                        scores=top_scores,
                        method=optimization_method,
                        risk_free_rate=risk_free_rate,
                        max_weight=max_weight,
                        min_weight=min_weight
                    )
                    
                    # Stocker les résultats dans la session
                    st.session_state.portfolio_weights = weights
                    st.session_state.portfolio_metrics = {
                        'expected_return': exp_return,
                        'expected_risk': exp_risk,
                        'sharpe_ratio': sharpe,
                        'details': details,
                        'method': optimization_method,
                        'top_tickers': top_tickers
                    }
                    st.session_state.optimization_done = True
                    
                    st.success("✅ Optimisation terminée avec succès!")
                except Exception as e:
                    st.error(f"Erreur lors de l'optimisation du portefeuille: {str(e)}")
        
        # Afficher les résultats si disponibles
        if 'optimization_done' in st.session_state and st.session_state.optimization_done:
            # Afficher les poids optimaux
            st.subheader("Allocation optimale du portefeuille")
            
            # Obtenir les poids et les trier
            weights = st.session_state.portfolio_weights
            sorted_weights = weights.sort_values(ascending=False)
            
            # Créer deux colonnes pour afficher les poids et le graphique
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Afficher les poids sous forme de tableau
                weights_df = pd.DataFrame({
                    'Poids (%)': sorted_weights * 100
                })
                
                st.dataframe(
                    weights_df.style.format({'Poids (%)': '{:.2f}'}),
                    use_container_width=True
                )
                
                # Afficher les métriques du portefeuille
                st.subheader("Métriques du portefeuille")
                
                metrics = {
                    'Rendement attendu (%)': st.session_state.portfolio_metrics['expected_return'] * 100,
                    'Risque attendu (%)': st.session_state.portfolio_metrics['expected_risk'] * 100,
                    'Ratio de Sharpe': st.session_state.portfolio_metrics['sharpe_ratio']
                }
                
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Métrique', 'Valeur'])
                
                st.dataframe(
                    metrics_df.style.format({'Valeur': '{:.2f}'}),
                    use_container_width=True
                )
                
                # Permettre de télécharger les poids
                st.download_button(
                    label="📥 Télécharger l'allocation",
                    data=weights_df.to_csv(index=True),
                    file_name=f"portfolio_allocation_{optimization_method}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Afficher le graphique en camembert des poids
                fig = px.pie(
                    values=sorted_weights.values,
                    names=sorted_weights.index,
                    title=f"Allocation du portefeuille ({optimization_method})"
                )
                
                # Personnaliser le graphique
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=500)
                
                # Afficher le graphique
                st.plotly_chart(fig, use_container_width=True)
            
            # Afficher la frontière efficiente (pour Markowitz)
            if optimization_method == "Markowitz" and 'efficient_frontier' in st.session_state.portfolio_metrics['details']:
                st.subheader("Frontière efficiente")
                
                # Extraire les données de la frontière efficiente
                ef_data = st.session_state.portfolio_metrics['details']['efficient_frontier']
                
                # Créer le graphique
                fig = go.Figure()
                
                # Ajouter la frontière efficiente
                fig.add_trace(go.Scatter(
                    x=ef_data['risks'],
                    y=ef_data['returns'],
                    mode='lines',
                    name='Frontière efficiente',
                    line=dict(color='blue')
                ))
                
                # Ajouter le portefeuille optimal
                fig.add_trace(go.Scatter(
                    x=[st.session_state.portfolio_metrics['expected_risk']],
                    y=[st.session_state.portfolio_metrics['expected_return']],
                    mode='markers',
                    name='Portefeuille optimal',
                    marker=dict(color='red', size=10)
                ))
                
                # Personnaliser le graphique
                fig.update_layout(
                    title="Frontière efficiente et portefeuille optimal",
                    xaxis_title="Risque (%)",
                    yaxis_title="Rendement (%)",
                    height=500
                )
                
                fig.update_xaxes(tickformat='.2%')
                fig.update_yaxes(tickformat='.2%')
                
                # Afficher le graphique
                st.plotly_chart(fig, use_container_width=True)
            
            # Afficher la relation entre les scores et les poids
            if optimization_method == "Score-Based":
                st.subheader("Relation scores vs poids")
                
                # Créer un DataFrame pour la comparaison
                compare_df = pd.DataFrame({
                    'Score': st.session_state.combined_scores.loc[weights.index, 'Combined_Score'],
                    'Poids': weights.values
                })
                
                # Créer le graphique de dispersion
                fig = px.scatter(
                    compare_df,
                    x='Score',
                    y='Poids',
                    text=weights.index,
                    title="Relation entre les scores et les poids attribués"
                )
                
                # Personnaliser le graphique
                fig.update_traces(
                    textposition='top center',
                    marker=dict(size=10)
                )
                
                fig.update_layout(
                    xaxis_title="Score combiné",
                    yaxis_title="Poids dans le portefeuille",
                    height=500
                )
                
                # Afficher le graphique
                st.plotly_chart(fig, use_container_width=True)

# Onglet 4: Backtesting
with tab4:
    st.header("📉 Backtesting du Portefeuille")
    st.markdown("Cette section permet de tester la performance historique du portefeuille optimisé.")
    
    # Vérifier si l'optimisation a été effectuée
    if 'optimization_done' not in st.session_state or not st.session_state.optimization_done:
        st.warning("⚠️ Veuillez d'abord optimiser le portefeuille dans l'onglet 'Optimisation'.")
    else:
        # Paramètres de backtesting
        st.subheader("Paramètres de backtesting")
        
        # Sélectionner la fréquence de rééquilibrage
        rebalance_freq = st.select_slider(
            "Fréquence de rééquilibrage",
            options=['D', 'W', 'M', 'Q', 'Y'],
            value='M',
            format_func=lambda x: {
                'D': 'Quotidien', 
                'W': 'Hebdomadaire', 
                'M': 'Mensuel', 
                'Q': 'Trimestriel', 
                'Y': 'Annuel'
            }[x]
        )
        
        # Sélectionner un benchmark
        benchmark_options = {
            "Equal Weight": "Équipondération des actions",
            "Market Index": "Indice de marché (S&P 500)"
        }
        
        benchmark_type = st.radio(
            "Benchmark pour la comparaison",
            options=list(benchmark_options.keys()),
            format_func=lambda x: benchmark_options[x],
            horizontal=True
        )
        
        # Bouton pour lancer le backtesting
        if st.button("🚀 Lancer le backtesting"):
            with st.spinner("Backtesting en cours..."):
                try:
                    # Récupérer les poids et les tickers
                    weights = st.session_state.portfolio_weights
                    top_tickers = st.session_state.portfolio_metrics['top_tickers']
                    
                    # Filtrer les données de prix pour les tickers sélectionnés
                    price_data = st.session_state.clean_price_data[top_tickers]
                    
                    # Effectuer le backtesting
                    performance, stats = backtest_portfolio(
                        price_data=price_data,
                        weights=weights,
                        rebalance_freq=rebalance_freq,
                        risk_free_rate=risk_free_rate
                    )
                    
                    # Créer un benchmark équipondéré
                    equal_weights = pd.Series(1/len(top_tickers), index=top_tickers)
                    benchmark_performance, benchmark_stats = backtest_portfolio(
                        price_data=price_data,
                        weights=equal_weights,
                        rebalance_freq=rebalance_freq,
                        risk_free_rate=risk_free_rate
                    )
                    
                    # Si l'utilisateur a choisi l'indice de marché comme benchmark
                    if benchmark_type == "Market Index":
                        # Récupérer les données de l'indice S&P 500
                        market_ticker = config['MARKET']['market_ticker']
                        market_data = st.session_state.price_data[[market_ticker]]
                        
                        # Normaliser à 100 pour la comparaison
                        market_perf = 100 * market_data / market_data.iloc[0].values[0]
                        market_perf.columns = ['Market Index']
                        
                        # Ajouter à la performance du benchmark
                        benchmark_performance['Portfolio_Value'] = market_perf['Market Index']
                    
                    # Stocker les résultats dans la session
                    st.session_state.backtest_results = {
                        'performance': performance,
                        'stats': stats,
                        'benchmark_performance': benchmark_performance,
                        'benchmark_stats': benchmark_stats,
                        'rebalance_freq': rebalance_freq,
                        'benchmark_type': benchmark_type
                    }
                    st.session_state.backtesting_done = True
                    
                    st.success("✅ Backtesting terminé avec succès!")
                except Exception as e:
                    st.error(f"Erreur lors du backtesting: {str(e)}")
        
        # Afficher les résultats si disponibles
        if 'backtesting_done' in st.session_state and st.session_state.backtesting_done:
            # Récupérer les résultats
            performance = st.session_state.backtest_results['performance']
            stats = st.session_state.backtest_results['stats']
            benchmark_performance = st.session_state.backtest_results['benchmark_performance']
            benchmark_stats = st.session_state.backtest_results['benchmark_stats']
            rebalance_freq = st.session_state.backtest_results['rebalance_freq']
            benchmark_type = st.session_state.backtest_results['benchmark_type']
            
            # Afficher les performances
            st.subheader("Performance du portefeuille vs Benchmark")
            
            # Normaliser à 100 pour la comparaison
            comparison = pd.DataFrame({
                f'Portefeuille ({st.session_state.portfolio_metrics["method"]})': performance['Portfolio_Value'],
                f'Benchmark ({benchmark_type})': benchmark_performance['Portfolio_Value']
            })
            
            comparison = 100 * comparison / comparison.iloc[0]
            
            # Créer le graphique de performance
            fig = px.line(
                comparison,
                x=comparison.index,
                y=comparison.columns,
                title="Performance du portefeuille vs Benchmark (base 100)"
            )
            
            # Personnaliser le graphique
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Valeur (base 100)",
                legend_title="Stratégie",
                height=500
            )
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les statistiques de performance
            st.subheader("Statistiques de performance")
            
            # Créer un DataFrame pour les statistiques
            stats_df = pd.DataFrame({
                f'Portefeuille ({st.session_state.portfolio_metrics["method"]})': [
                    stats['Cumulative_Return'] * 100,
                    stats['Annual_Return'] * 100,
                    stats['Annual_Volatility'] * 100,
                    stats['Sharpe_Ratio'],
                    stats['Max_Drawdown'] * 100,
                    stats['Win_Rate'] * 100
                ],
                f'Benchmark ({benchmark_type})': [
                    benchmark_stats['Cumulative_Return'] * 100,
                    benchmark_stats['Annual_Return'] * 100,
                    benchmark_stats['Annual_Volatility'] * 100,
                    benchmark_stats['Sharpe_Ratio'],
                    benchmark_stats['Max_Drawdown'] * 100,
                    benchmark_stats['Win_Rate'] * 100
                ]
            }, index=[
                'Rendement cumulatif (%)',
                'Rendement annualisé (%)',
                'Volatilité annualisée (%)',
                'Ratio de Sharpe',
                'Drawdown maximal (%)',
                'Taux de succès (%)'
            ])
            
            # Afficher le tableau de statistiques
            st.dataframe(
                stats_df.style.format('{:.2f}').background_gradient(subset=[f'Portefeuille ({st.session_state.portfolio_metrics["method"]})'], cmap='viridis'),
                use_container_width=True
            )
            
            # Afficher le drawdown
            st.subheader("Analyse du Drawdown")
            
            # Calculer le drawdown pour le portefeuille et le benchmark
            portfolio_dd = (performance['Portfolio_Value'] / performance['Portfolio_Value'].cummax() - 1) * 100
            benchmark_dd = (benchmark_performance['Portfolio_Value'] / benchmark_performance['Portfolio_Value'].cummax() - 1) * 100
            
            # Créer un DataFrame pour le drawdown
            dd_df = pd.DataFrame({
                f'Portefeuille ({st.session_state.portfolio_metrics["method"]})': portfolio_dd,
                f'Benchmark ({benchmark_type})': benchmark_dd
            })
            
            # Créer le graphique de drawdown
            fig = px.line(
                dd_df,
                x=dd_df.index,
                y=dd_df.columns,
                title="Drawdown au cours du temps"
            )
            
            # Personnaliser le graphique
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                legend_title="Stratégie",
                height=400,
                yaxis=dict(autorange="reversed")  # Inverser l'axe y pour que le drawdown soit vers le bas
            )
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les rendements mensuels
            st.subheader("Rendements mensuels")
            
            # Calculer les rendements mensuels
            monthly_returns = performance['Portfolio_Value'].resample('M').last().pct_change() * 100
            monthly_returns = monthly_returns.dropna()
            
            # Créer un heatmap des rendements mensuels
            monthly_returns_pivot = monthly_returns.copy()
            monthly_returns_pivot.index = pd.MultiIndex.from_arrays(
                [monthly_returns.index.year, monthly_returns.index.month],
                names=['Année', 'Mois']
            )
            monthly_returns_pivot = monthly_returns_pivot.reset_index()
            monthly_returns_pivot = monthly_returns_pivot.pivot(index='Année', columns='Mois', values='Portfolio_Value')
            monthly_returns_pivot.columns = [pd.to_datetime(f"2000-{m}-1").strftime('%b') for m in monthly_returns_pivot.columns]
            
            # Créer le heatmap avec Plotly
            fig = px.imshow(
                monthly_returns_pivot,
                labels=dict(x="Mois", y="Année", color="Rendement (%)"),
                x=monthly_returns_pivot.columns,
                y=monthly_returns_pivot.index,
                color_continuous_scale="RdYlGn",  # Rouge pour négatif, vert pour positif
                title="Heatmap des rendements mensuels (%)"
            )
            
            # Ajouter les valeurs dans les cellules
            for i, year in enumerate(monthly_returns_pivot.index):
                for j, month in enumerate(monthly_returns_pivot.columns):
                    value = monthly_returns_pivot.iloc[i, j]
                    if not pd.isna(value):
                        fig.add_annotation(
                            x=month,
                            y=year,
                            text=f"{value:.1f}%",
                            showarrow=False,
                            font=dict(color="black" if abs(value) < 10 else "white")
                        )
            
            # Personnaliser le graphique
            fig.update_layout(height=400)
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
            
            # Permettre de télécharger les résultats du backtest
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Télécharger les performances",
                    data=comparison.to_csv(index=True),
                    file_name="backtest_performance.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="📥 Télécharger les statistiques",
                    data=stats_df.to_csv(index=True),
                    file_name="backtest_statistics.csv",
                    mime="text/csv"
                )

# Ajouter des informations dans la sidebar
with st.sidebar:
    st.markdown("---")
    st.caption("© 2025 Stock Picking Model | v1.0.0")
    st.caption("Développé avec Streamlit et Python")
    
    # Ajouter un lien vers le GitHub
    st.markdown("[Code source sur GitHub](https://github.com/Kyac99/stock-picking-model)")
