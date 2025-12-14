"""
Application Streamlit - Niveau 3 : Robustesse et Analyse des Limites
À INTÉGRER dans app.py ou créer un fichier séparé
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Ajoute le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import prepare_data
from pareto_metrics import (
    calculate_all_metrics, compare_fronts, hypervolume_2d, 
    spacing_metric, spread_metric
)
from robustness import (
    black_litterman, bootstrap_resampling, optimize_with_bootstrap,
    sensitivity_analysis, resampled_efficient_frontier,
    calculate_var_cvar, create_view_matrix
)
from limits_analysis import (
    test_normality, plot_qq_plots, test_stationarity,
    rolling_statistics, plot_rolling_statistics,
    correlation_stability, plot_correlation_stability,
    estimation_error_analysis, concentration_analysis,
    generate_limits_report
)
from optimizers.classic import (
    generate_efficient_frontier_scalarization,
    find_tangency_portfolio
)
from optimizers.genetic import optimize_nsga2_biobjective, extract_pareto_front

# Configuration
st.set_page_config(page_title="Niveau 3 - Robustesse", page_icon="🔬", layout="wide")

# === FONCTION POUR AJOUTER DANS app.py ===

def page_niveau3():
    """
    Page dédiée au Niveau 3 : Robustesse et Analyse des Limites
    """
    st.title("🔬 Niveau 3 : Robustesse et Analyse des Limites")
    st.markdown("---")
    
    # Chargement des données
    @st.cache_data
    def load_data():
        prices, returns, mu, Sigma, ticker_sectors = prepare_data(
            start_date="2020-01-01",
            end_date="2024-12-31",
            data_dir="data/raw"
        )
        tickers = list(prices.columns)
        return prices, returns, mu, Sigma, tickers, ticker_sectors
    
    with st.spinner("Chargement des données..."):
        prices, returns, mu, Sigma, tickers, ticker_sectors = load_data()
    
    st.sidebar.success(f"✅ {len(tickers)} actifs chargés")
    
    # Sélection de l'analyse
    analysis_type = st.sidebar.selectbox(
        "Type d'analyse",
        [
            "📊 Comparaison Quantitative des Méthodes",
            "🔍 Analyse des Limites du Modèle",
            "🛡️ Optimisation Robuste (Black-Litterman)",
            "🔄 Rééchantillonnage Bootstrap",
            "📈 Analyse de Sensibilité",
            "📉 VaR et CVaR"
        ]
    )
    
    # === COMPARAISON QUANTITATIVE ===
    if "Comparaison Quantitative" in analysis_type:
        st.header("📊 Comparaison Quantitative des Méthodes")
        
        st.info("Compare les fronts de Pareto obtenus par différentes méthodes avec des indicateurs de qualité.")
        
        col1, col2 = st.columns(2)
        with col1:
            n_points = st.slider("Nombre de points", 20, 100, 50)
        with col2:
            n_gen = st.slider("Générations NSGA-II", 50, 200, 100)
        
        if st.button("🚀 Lancer la Comparaison", key="btn_compare_metrics"):
            with st.spinner("Optimisation en cours..."):
                # Méthode 1 : Scalarisation
                portfolios_scal, rets_scal, risks_scal = generate_efficient_frontier_scalarization(
                    mu, Sigma, n_points=n_points
                )
                F_scal = np.column_stack([-rets_scal, risks_scal])
                
                # Méthode 2 : NSGA-II
                res_nsga = optimize_nsga2_biobjective(mu, Sigma, K=None, 
                                                      pop_size=100, n_gen=n_gen)
                X_nsga, F_nsga = extract_pareto_front(res_nsga)
                
                st.success("✅ Optimisation terminée!")
                
                # Calcul des métriques
                st.subheader("Indicateurs de Qualité du Front de Pareto")
                
                metrics_df = compare_fronts(
                    [F_scal, F_nsga],
                    ['Scalarisation', 'NSGA-II']
                )
                
                # Affichage formaté
                st.dataframe(
                    metrics_df.style.format({
                        'n_solutions': '{:.0f}',
                        'hypervolume': '{:.6f}',
                        'spacing': '{:.6f}',
                        'spread': '{:.4f}',
                        'gd': '{:.6f}',
                        'igd': '{:.6f}',
                        'epsilon': '{:.6f}'
                    }).background_gradient(subset=['hypervolume', 'spread'], cmap='Greens')
                      .background_gradient(subset=['spacing', 'gd', 'igd', 'epsilon'], cmap='Reds_r'),
                    use_container_width=True
                )
                
                # Interprétation
                st.markdown("""
                **Interprétation des indicateurs** :
                - **Hypervolume** : Volume de l'espace dominé (↑ meilleur)
                - **Spacing** : Uniformité de la distribution (↓ meilleur)
                - **Spread** : Étendue du front (↑ meilleur)
                - **GD/IGD** : Distance au front de référence (↓ meilleur)
                - **Epsilon** : Facteur d'éloignement minimal (↓ meilleur)
                """)
                
                # Visualisation comparative
                st.subheader("Comparaison Visuelle des Fronts")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=np.sqrt(risks_scal) * 100,
                    y=rets_scal * 100,
                    mode='markers',
                    name='Scalarisation',
                    marker=dict(size=8, color='blue', symbol='circle')
                ))
                
                rets_nsga = -F_nsga[:, 0]
                risks_nsga = F_nsga[:, 1]
                
                fig.add_trace(go.Scatter(
                    x=np.sqrt(risks_nsga) * 100,
                    y=rets_nsga * 100,
                    mode='markers',
                    name='NSGA-II',
                    marker=dict(size=8, color='red', symbol='diamond')
                ))
                
                fig.update_layout(
                    title="Comparaison des Fronts de Pareto",
                    xaxis_title="Volatilité (%)",
                    yaxis_title="Rendement (%)",
                    height=600,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse détaillée
                st.subheader("Analyse Détaillée")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_hv = metrics_df.loc[metrics_df['hypervolume'].idxmax(), 'name']
                    st.metric("Meilleur Hypervolume", best_hv)
                
                with col2:
                    best_spacing = metrics_df.loc[metrics_df['spacing'].idxmin(), 'name']
                    st.metric("Meilleur Spacing", best_spacing)
                
                with col3:
                    best_spread = metrics_df.loc[metrics_df['spread'].idxmax(), 'name']
                    st.metric("Meilleur Spread", best_spread)
                
                # Conclusion
                st.success(f"""
                **Conclusion** : 
                - La méthode **{best_hv}** génère le front avec le plus grand hypervolume (meilleure couverture)
                - La méthode **{best_spacing}** produit les solutions les plus uniformément espacées
                - La méthode **{best_spread}** couvre la plus grande étendue de l'espace objectif
                """)
    
    # === ANALYSE DES LIMITES ===
    elif "Analyse des Limites" in analysis_type:
        st.header("🔍 Analyse des Limites du Modèle")
        
        tabs = st.tabs([
            "📉 Normalité", 
            "📊 Stationnarité", 
            "🔗 Stabilité Corrélations",
            "❌ Erreurs Estimation",
            "🎯 Concentration"
        ])
        
        # TAB 1 : Normalité
        with tabs[0]:
            st.subheader("Test de Normalité des Rendements")
            
            st.info("L'hypothèse de normalité de Markowitz est souvent violée. Analysons...")
            
            if st.button("🧪 Tester la Normalité"):
                with st.spinner("Tests en cours..."):
                    norm_results = test_normality(returns)
                    
                    st.markdown("### Résultats des Tests Statistiques")
                    
                    # Résumé
                    n_total = len(norm_results)
                    n_reject_jb = norm_results['JB_reject'].sum()
                    n_reject_sw = norm_results['SW_reject'].sum()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Actifs", n_total)
                    col2.metric("Rejet Jarque-Bera", f"{n_reject_jb} ({n_reject_jb/n_total*100:.1f}%)")
                    col3.metric("Rejet Shapiro-Wilk", f"{n_reject_sw} ({n_reject_sw/n_total*100:.1f}%)")
                    
                    # Tableau détaillé
                    st.dataframe(
                        norm_results.style.format({
                            'JB_stat': '{:.2f}',
                            'JB_pvalue': '{:.4f}',
                            'SW_stat': '{:.4f}',
                            'SW_pvalue': '{:.4f}',
                            'Skewness': '{:.3f}',
                            'Kurtosis': '{:.3f}'
                        }).applymap(lambda x: 'background-color: #ffcccc' if x else '', 
                                   subset=['JB_reject', 'SW_reject']),
                        use_container_width=True,
                        height=400
                    )
                    
                    # QQ-Plots
                    st.markdown("### QQ-Plots (9 premiers actifs)")
                    fig_qq = plot_qq_plots(returns, n_plots=9)
                    st.pyplot(fig_qq)
                    
                    # Conclusion
                    if n_reject_jb > n_total * 0.5:
                        st.warning(f"""
                        ⚠️ **Alerte** : {n_reject_jb/n_total*100:.1f}% des actifs rejettent l'hypothèse de normalité.
                        
                        **Implications** :
                        - Le risque de Markowitz (variance) sous-estime le risque extrême
                        - Les probabilités de krach sont sous-estimées
                        - Solutions : Utiliser CVaR, distributions à queues épaisses (Student-t)
                        """)
                    else:
                        st.success("✅ La majorité des actifs suivent approximativement une loi normale.")
        
        # TAB 2 : Stationnarité
        with tabs[1]:
            st.subheader("Test de Stationnarité (ADF)")
            
            if st.button("🧪 Tester la Stationnarité"):
                with st.spinner("Tests en cours..."):
                    stat_results = test_stationarity(returns)
                    
                    n_stationary = stat_results['Stationary'].sum()
                    pct_stationary = n_stationary / len(stat_results) * 100
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Actifs Stationnaires", n_stationary)
                    col2.metric("Pourcentage", f"{pct_stationary:.1f}%")
                    
                    st.dataframe(
                        stat_results.style.format({
                            'ADF_stat': '{:.4f}',
                            'ADF_pvalue': '{:.4f}',
                            'Critical_1%': '{:.4f}',
                            'Critical_5%': '{:.4f}',
                            'Critical_10%': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Statistiques glissantes pour un actif
                    st.markdown("### Statistiques Glissantes (Exemple)")
                    selected_asset = st.selectbox("Choisir un actif", tickers)
                    
                    fig_rolling = plot_rolling_statistics(returns, selected_asset, window=252)
                    st.pyplot(fig_rolling)
                    
                    st.info("""
                    **Observation** : Les rendements moyens et volatilités varient dans le temps.
                    Cela viole l'hypothèse de stationnarité de Markowitz.
                    """)
        
        # TAB 3 : Stabilité des Corrélations
        with tabs[2]:
            st.subheader("Stabilité des Corrélations dans le Temps")
            
            n_periods = st.slider("Nombre de périodes", 3, 10, 5)
            
            if st.button("📊 Analyser la Stabilité"):
                with st.spinner("Analyse en cours..."):
                    corr_stab = correlation_stability(returns, n_periods=n_periods)
                    
                    st.metric("Instabilité Moyenne", f"{corr_stab['mean_stability']:.4f}")
                    
                    fig_corr = plot_correlation_stability(corr_stab)
                    st.pyplot(fig_corr)
                    
                    if corr_stab['mean_stability'] > 0.2:
                        st.warning(f"""
                        ⚠️ **Corrélations instables** (σ = {corr_stab['mean_stability']:.3f})
                        
                        Les corrélations varient significativement dans le temps, notamment en période de crise.
                        Cela peut rendre les portefeuilles optimisés obsolètes rapidement.
                        """)
                    else:
                        st.success("✅ Les corrélations sont relativement stables.")
        
        # TAB 4 : Erreurs d'Estimation
        with tabs[3]:
            st.subheader("Erreurs d'Estimation (In-Sample vs Out-of-Sample)")
            
            train_size = st.slider("Proportion d'entraînement", 0.5, 0.9, 0.8, 0.05)
            n_iter = st.slider("Nombre d'itérations", 10, 200, 50)
            
            if st.button("🔄 Analyser les Erreurs"):
                with st.spinner(f"Analyse sur {n_iter} itérations..."):
                    errors = estimation_error_analysis(returns, train_size=train_size, 
                                                      n_iterations=n_iter)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Erreur sur μ")
                        st.metric("Erreur Moyenne", f"{errors['mu_error_mean']:.4f}")
                        st.metric("Écart-type", f"{errors['mu_error_std']:.4f}")
                    
                    with col2:
                        st.markdown("### Erreur sur Σ")
                        st.metric("Erreur Moyenne", f"{errors['sigma_error_mean']:.6f}")
                        st.metric("Écart-type", f"{errors['sigma_error_std']:.6f}")
                    
                    # Histogrammes
                    fig = make_subplots(rows=1, cols=2, 
                                       subplot_titles=("Erreurs sur μ", "Erreurs sur Σ"))
                    
                    fig.add_trace(
                        go.Histogram(x=errors['mu_errors'], nbinsx=30, name='μ'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Histogram(x=errors['sigma_errors'], nbinsx=30, name='Σ'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.warning("""
                    ⚠️ **Impact** : Les erreurs d'estimation peuvent conduire à des portefeuilles très différents.
                    Une petite erreur sur μ peut changer drastiquement les poids optimaux.
                    """)
        
        # TAB 5 : Concentration
        with tabs[4]:
            st.subheader("Analyse de Concentration du Portefeuille")
            
            st.info("Calculons la concentration d'un portefeuille optimal...")
            
            if st.button("📈 Analyser la Concentration"):
                with st.spinner("Optimisation et analyse..."):
                    w_tangent = find_tangency_portfolio(mu, Sigma, rf=0.02)
                    
                    conc = concentration_analysis(w_tangent, tickers, ticker_sectors)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Herfindahl (Actifs)", f"{conc['herfindahl_assets']:.4f}")
                    col1.metric("N Effectif Actifs", f"{conc['effective_n_assets']:.1f}")
                    
                    col2.metric("Herfindahl (Secteurs)", f"{conc['herfindahl_sectors']:.4f}")
                    col2.metric("N Effectif Secteurs", f"{conc['effective_n_sectors']:.1f}")
                    
                    col3.metric("Top 5 Concentration", f"{conc['top5_concentration']*100:.1f}%")
                    col3.metric("Top 10 Concentration", f"{conc['top10_concentration']*100:.1f}%")
                    
                    # Camembert sectoriel
                    sector_df = pd.DataFrame(
                        list(conc['sector_weights'].items()),
                        columns=['Secteur', 'Poids']
                    )
                    sector_df['Poids (%)'] = sector_df['Poids'] * 100
                    
                    fig = px.pie(sector_df, values='Poids (%)', names='Secteur',
                                title='Répartition Sectorielle')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if conc['top5_concentration'] > 0.6:
                        st.warning(f"""
                        ⚠️ **Forte concentration** : Les 5 principaux actifs représentent {conc['top5_concentration']*100:.1f}% du portefeuille.
                        
                        Cela augmente le risque idiosyncratique et peut poser des problèmes de liquidité.
                        Solution : Ajouter des contraintes de diversification (cardinalité, poids max).
                        """)
    
    # === BLACK-LITTERMAN ===
    elif "Black-Litterman" in analysis_type:
        st.header("🛡️ Optimisation Robuste : Black-Litterman")
        
        st.markdown("""
        Le modèle de Black-Litterman combine :
        - Les **rendements d'équilibre du marché** (prudents)
        - Vos **vues subjectives** sur certains actifs (expertise)
        
        Cela réduit l'impact des erreurs d'estimation de μ.
        """)
        
        # Sélection des vues
        st.subheader("Définir vos Vues")
        
        n_views = st.number_input("Nombre de vues", 1, 5, 1)
        
        P_list = []
        Q_list = []
        
        for i in range(n_views):
            st.markdown(f"**Vue {i+1}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                view_type = st.selectbox(f"Type", ['absolute', 'relative'], key=f"type_{i}")
            
            with col2:
                asset1 = st.selectbox(f"Actif", tickers, key=f"asset1_{i}")
                asset1_idx = tickers.index(asset1)
            
            with col3:
                if view_type == 'relative':
                    asset2 = st.selectbox(f"vs Actif", tickers, key=f"asset2_{i}")
                    asset2_idx = tickers.index(asset2)
                else:
                    asset2_idx = None
            
            expected_return = st.slider(
                f"Rendement attendu (%)",
                -20.0, 50.0, 10.0, 0.5,
                key=f"return_{i}"
            ) / 100
            
            P = create_view_matrix(len(tickers), view_type, [asset1_idx], asset2_idx)[0]
            P_list.append(P)
            Q_list.append(expected_return)
        
        P_matrix = np.vstack(P_list)
        Q_vector = np.array(Q_list)
        
        if st.button("🚀 Optimiser avec Black-Litterman"):
            with st.spinner("Optimisation en cours..."):
                # Black-Litterman
                mu_bl, Sigma_bl = black_litterman(
                    Sigma, 
                    market_caps=None,  # Équipondéré par défaut
                    P=P_matrix,
                    Q=Q_vector,
                    tau=0.025
                )
                
                # Optimisation classique
                w_classic = find_tangency_portfolio(mu, Sigma, rf=0.02)
                
                # Optimisation BL
                w_bl = find_tangency_portfolio(mu_bl, Sigma_bl, rf=0.02)
                
                # Comparaison
                st.subheader("Comparaison : Classique vs Black-Litterman")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Portefeuille Classique")
                    ret_classic = w_classic @ mu
                    risk_classic = np.sqrt(w_classic @ Sigma @ w_classic)
                    st.metric("Rendement", f"{ret_classic*100:.2f}%")
                    st.metric("Volatilité", f"{risk_classic*100:.2f}%")
                    st.metric("Sharpe", f"{(ret_classic-0.02)/risk_classic:.3f}")
                
                with col2:
                    st.markdown("### Portefeuille Black-Litterman")
                    ret_bl = w_bl @ mu_bl
                    risk_bl = np.sqrt(w_bl @ Sigma_bl @ w_bl)
                    st.metric("Rendement", f"{ret_bl*100:.2f}%")
                    st.metric("Volatilité", f"{risk_bl*100:.2f}%")
                    st.metric("Sharpe", f"{(ret_bl-0.02)/risk_bl:.3f}")
                
                # Changements de poids
                st.subheader("Changements de Poids")
                
                weight_changes = pd.DataFrame({
                    'Ticker': tickers,
                    'Classique': w_classic * 100,
                    'Black-Litterman': w_bl * 100,
                    'Différence': (w_bl - w_classic) * 100
                })
                
                # Filtre les changements significatifs
                weight_changes = weight_changes[np.abs(weight_changes['Différence']) > 0.1]
                weight_changes = weight_changes.sort_values('Différence', ascending=False)
                
                st.dataframe(
                    weight_changes.style.format({
                        'Classique': '{:.2f}%',
                        'Black-Litterman': '{:.2f}%',
                        'Différence': '{:+.2f}%'
                    }).background_gradient(subset=['Différence'], cmap='RdYlGn', vmin=-10, vmax=10),
                    use_container_width=True
                )
                
                st.success("""
                ✅ **Avantage** : Black-Litterman intègre vos convictions tout en restant prudent.
                Les poids sont plus stables et moins sensibles aux erreurs d'estimation.
                """)
    
    # === BOOTSTRAP ===
    elif "Bootstrap" in analysis_type:
        st.header("🔄 Rééchantillonnage Bootstrap")
        
        st.markdown("""
        Le bootstrap permet d'évaluer la **stabilité** des portefeuilles optimaux.
        On génère 100+ échantillons et on optimise sur chacun.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            n_bootstrap = st.slider("Nombre d'échantillons", 10, 200, 50)
        with col2:
            block_size = st.slider("Taille des blocs", 5, 60, 20)
        
        if st.button("🔄 Lancer le Bootstrap"):
            with st.spinner(f"Génération de {n_bootstrap} échantillons..."):
                # Bootstrap
                bootstrap_samples = bootstrap_resampling(
                    returns, 
                    n_samples=n_bootstrap,
                    block_size=block_size
                )
                
                # Optimisation sur chaque échantillon
                from optimizers.classic import find_tangency_portfolio
                
                w_mean, w_std = optimize_with_bootstrap(
                    mu, Sigma, bootstrap_samples,
                    optimizer_func=find_tangency_portfolio,
                    rf=0.02
                )
                
                st.success("✅ Bootstrap terminé!")
                
                # Résultats
                st.subheader("Portefeuille Moyen (Bootstrap)")
                
                col1, col2 = st.columns(2)
                
                ret_mean = w_mean @ mu
                risk_mean = np.sqrt(w_mean @ Sigma @ w_mean)
                
                col1.metric("Rendement", f"{ret_mean*100:.2f}%")
                col2.metric("Volatilité", f"{risk_mean*100:.2f}%")
                
                # Top positions avec intervalle de confiance
                st.subheader("Top 10 Positions (avec IC 95%)")
                
                top_indices = np.argsort(w_mean)[-10:][::-1]
                
                bootstrap_df = pd.DataFrame({
                    'Ticker': [tickers[i] for i in top_indices],
                    'Poids Moyen (%)': [w_mean[i] * 100 for i in top_indices],
                    'Écart-type (%)': [w_std[i] * 100 for i in top_indices],
                    'IC Inf (%)': [(w_mean[i] - 1.96*w_std[i]) * 100 for i in top_indices],
                    'IC Sup (%)': [(w_mean[i] + 1.96*w_std[i]) * 100 for i in top_indices]
                })
                
                st.dataframe(
                    bootstrap_df.style.format({
                        'Poids Moyen (%)': '{:.2f}',
                        'Écart-type (%)': '{:.2f}',
                        'IC Inf (%)': '{:.2f}',
                        'IC Sup (%)': '{:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Graphique des incertitudes
                fig = go.Figure()
                
                for i, idx in enumerate(top_indices):
                    fig.add_trace(go.Box(
                        y=[w_mean[idx] * 100],
                        error_y=dict(
                            type='data',
                            array=[1.96 * w_std[idx] * 100],
                            arrayminus=[1.96 * w_std[idx] * 100]
                        ),
                        name=tickers[idx],
                        marker_color='steelblue'
                    ))
                
                fig.update_layout(
                    title="Incertitude des Poids (IC 95%)",
                    yaxis_title="Poids (%)",
                    showlegend=False,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interprétation
                max_std_idx = np.argmax([w_std[i] for i in top_indices])
                max_std_ticker = tickers[top_indices[max_std_idx]]
                max_std_value = w_std[top_indices[max_std_idx]] * 100
                
                st.info(f"""
                **Observation** : 
                - L'actif **{max_std_ticker}** a la plus grande incertitude (σ = {max_std_value:.2f}%)
                - Cela indique que son poids optimal varie beaucoup selon les échantillons
                - Solution : Privilégier les actifs avec faible σ pour un portefeuille stable
                """)
    
    # === SENSIBILITÉ ===
    elif "Sensibilité" in analysis_type:
        st.header("📈 Analyse de Sensibilité")
        
        st.markdown("""
        Comment le portefeuille optimal change-t-il si on perturbe les paramètres ?
        """)
        
        param_to_perturb = st.selectbox("Paramètre à perturber", ['mu', 'Sigma'])
        perturbation_pct = st.slider("Amplitude de perturbation (%)", 1, 20, 10)
        
        if st.button("🔍 Analyser la Sensibilité"):
            with st.spinner("Analyse en cours..."):
                from optimizers.classic import find_tangency_portfolio
                
                perturbation_range = np.linspace(
                    -perturbation_pct/100,
                    perturbation_pct/100,
                    21
                )
                
                sens_results = sensitivity_analysis(
                    mu, Sigma,
                    optimizer_func=find_tangency_portfolio,
                    param_name=param_to_perturb,
                    perturbation_range=perturbation_range,
                    rf=0.02
                )
                
                # Graphiques
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        f"Rendement vs Perturbation de {param_to_perturb}",
                        f"Risque vs Perturbation de {param_to_perturb}",
                        f"Changement Total des Poids",
                        "Quelques Actifs Individuels"
                    )
                )
                
                perturbations_pct = [p*100 for p in sens_results['perturbations']]
                
                # Rendement
                fig.add_trace(
                    go.Scatter(x=perturbations_pct, 
                              y=[r*100 for r in sens_results['returns']],
                              mode='lines+markers', name='Rendement'),
                    row=1, col=1
                )
                
                # Risque
                fig.add_trace(
                    go.Scatter(x=perturbations_pct,
                              y=[np.sqrt(r)*100 for r in sens_results['risks']],
                              mode='lines+markers', name='Volatilité'),
                    row=1, col=2
                )
                
                # Changement total
                fig.add_trace(
                    go.Scatter(x=perturbations_pct,
                              y=sens_results['weight_changes'],
                              mode='lines+markers', name='Δ Poids Total',
                              line=dict(color='red')),
                    row=2, col=1
                )
                
                # Quelques actifs individuels
                w_ref = sens_results['weights'][len(sens_results['weights'])//2]
                top_assets = np.argsort(w_ref)[-3:]
                
                for idx in top_assets:
                    asset_weights = [w[idx]*100 for w in sens_results['weights']]
                    fig.add_trace(
                        go.Scatter(x=perturbations_pct, y=asset_weights,
                                  mode='lines', name=tickers[idx]),
                        row=2, col=2
                    )
                
                fig.update_xaxes(title_text=f"Perturbation de {param_to_perturb} (%)", row=2, col=1)
                fig.update_xaxes(title_text=f"Perturbation de {param_to_perturb} (%)", row=2, col=2)
                fig.update_yaxes(title_text="Rendement (%)", row=1, col=1)
                fig.update_yaxes(title_text="Volatilité (%)", row=1, col=2)
                fig.update_yaxes(title_text="Changement", row=2, col=1)
                fig.update_yaxes(title_text="Poids (%)", row=2, col=2)
                
                fig.update_layout(height=800, showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mesure de sensibilité
                max_change = max(sens_results['weight_changes'])
                st.metric("Changement Maximum des Poids", f"{max_change:.3f}")
                
                if max_change > 0.5:
                    st.warning(f"""
                    ⚠️ **Haute sensibilité** : Une perturbation de ±{perturbation_pct}% sur {param_to_perturb} 
                    change les poids de jusqu'à {max_change:.1%}.
                    
                    **Implication** : Le portefeuille est très sensible aux erreurs d'estimation.
                    Solution : Utiliser Black-Litterman, bootstrap, ou contraintes de stabilité.
                    """)
                else:
                    st.success("✅ Le portefeuille est relativement robuste aux perturbations.")
    
    # === VAR / CVAR ===
    else:  # VaR et CVaR
        st.header("📉 Value-at-Risk et Conditional VaR")
        
        st.markdown("""
        La variance de Markowitz ne capture pas le risque extrême (queues de distribution).
        **VaR et CVaR** mesurent les pertes potentielles dans les pires scénarios.
        """)
        
        confidence = st.slider("Niveau de confiance (%)", 90, 99, 95) / 100
        
        if st.button("📊 Calculer VaR et CVaR"):
            with st.spinner("Calcul en cours..."):
                # Optimise un portefeuille
                w_tangent = find_tangency_portfolio(mu, Sigma, rf=0.02)
                
                # Calcule VaR et CVaR
                var, cvar = calculate_var_cvar(returns, w_tangent, confidence=confidence)
                
                # Affichage
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Rendement Espéré", f"{(w_tangent@mu)*100:.2f}%")
                col2.metric(f"VaR ({confidence*100:.0f}%)", f"{var:.2f}%", 
                           delta=None, delta_color="inverse")
                col3.metric(f"CVaR ({confidence*100:.0f}%)", f"{cvar:.2f}%",
                           delta=None, delta_color="inverse")
                
                # Distribution des rendements du portefeuille
                portfolio_returns = (returns * w_tangent).sum(axis=1) * 100
                
                fig = go.Figure()
                
                # Histogramme
                fig.add_trace(go.Histogram(
                    x=portfolio_returns,
                    nbinsx=50,
                    name='Distribution',
                    marker_color='steelblue',
                    opacity=0.7
                ))
                
                # Ligne VaR
                fig.add_vline(x=-var, line_color='red', line_dash='dash',
                             annotation_text=f'VaR ({confidence*100:.0f}%)',
                             annotation_position='top right')
                
                # Ligne CVaR (zone)
                fig.add_vrect(
                    x0=portfolio_returns.min(),
                    x1=-var,
                    fillcolor='red',
                    opacity=0.2,
                    annotation_text=f'CVaR = {cvar:.2f}%',
                    annotation_position='inside top left'
                )
                
                fig.update_layout(
                    title="Distribution des Rendements Quotidiens du Portefeuille",
                    xaxis_title="Rendement (%)",
                    yaxis_title="Fréquence",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interprétation
                st.markdown(f"""
                **Interprétation** :
                - **VaR** : Dans {(1-confidence)*100:.1f}% des cas, les pertes dépasseront {var:.2f}%
                - **CVaR** : Quand les pertes dépassent la VaR, la perte moyenne est de {cvar:.2f}%
                - Le ratio Rendement/CVaR = {(w_tangent@mu)*100/cvar:.2f}
                
                **Comparaison** :
                - Volatilité (Markowitz) = {np.sqrt(w_tangent @ Sigma @ w_tangent)*100:.2f}%
                - CVaR (mesure de risque extrême) = {cvar:.2f}%
                
                Si CVaR >> Volatilité, cela indique des queues épaisses (risque de krach).
                """)
                
                if cvar > np.sqrt(w_tangent @ Sigma @ w_tangent) * 100 * 1.5:
                    st.warning("""
                    ⚠️ **Alerte** : CVaR significativement supérieure à la volatilité.
                    Les rendements ont des queues épaisses. Le risque de krach est sous-estimé par Markowitz.
                    """)


# === À AJOUTER DANS LE FICHIER app.py PRINCIPAL ===

# Remplacez la section "Comparaison" par ceci :

# elif "Comparaison" in niveau:
#     page_niveau3()  # Appelle la nouvelle page

if __name__ == "__main__":
    page_niveau3()