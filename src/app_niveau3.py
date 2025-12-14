"""
Application Streamlit - Niveau 3 : Robustesse et Analyse des Limites
VERSION STRICTE CONFORME AU PDF (Sans Black-Litterman ni VaR)
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
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import prepare_data
from pareto_metrics import (
    compare_fronts
)
from robustness import (
    bootstrap_resampling, optimize_with_bootstrap
)
from limits_analysis import (
    test_normality, plot_qq_plots, test_stationarity,
    plot_rolling_statistics, correlation_stability,
    plot_correlation_stability, estimation_error_analysis, 
    concentration_analysis
)
from optimizers.classic import (
    generate_efficient_frontier_scalarization,
    find_tangency_portfolio
)
from optimizers.genetic import optimize_nsga2_biobjective, extract_pareto_front

# Configuration de la page (si lancé seul)
# st.set_page_config(page_title="Niveau 3 - Robustesse", page_icon="🔬", layout="wide")

def page_niveau3():
    """
    Page dédiée au Niveau 3 : Robustesse et Analyse des Limites
    Conforme aux exigences strictes du PDF.
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
    
    # Sélection de l'analyse (OPTIONS RÉDUITES AU STRICT NÉCESSAIRE)
    analysis_type = st.sidebar.selectbox(
        "Type d'analyse",
        [
            "📊 Comparaison des Méthodes (Markowitz vs NSGA-II)",
            "🔄 Robustesse par Rééchantillonnage (Bootstrap)",
            "🔍 Analyse des Limites du Modèle"
        ]
    )
    
    # === 1. COMPARAISON QUANTITATIVE ===
    if "Comparaison" in analysis_type:
        st.header("📊 Comparaison : Scalarisation vs NSGA-II")
        st.info("Comparaison du modèle classique (convexe) et de l'approche heuristique (NSGA-II) utilisée pour la cardinalité.")
        
        col1, col2 = st.columns(2)
        with col1:
            n_points = st.slider("Nombre de points (Frontière)", 20, 100, 50)
        with col2:
            n_gen = st.slider("Générations NSGA-II", 50, 200, 100)
        
        if st.button("🚀 Lancer la Comparaison"):
            with st.spinner("Calcul des frontières..."):
                # Méthode 1 : Scalarisation (Benchmark)
                portfolios_scal, rets_scal, risks_scal = generate_efficient_frontier_scalarization(
                    mu, Sigma, n_points=n_points
                )
                F_scal = np.column_stack([-rets_scal, risks_scal])
                
                # Méthode 2 : NSGA-II
                res_nsga = optimize_nsga2_biobjective(mu, Sigma, K=None, 
                                                      pop_size=100, n_gen=n_gen)
                X_nsga, F_nsga = extract_pareto_front(res_nsga)
                
                st.success("✅ Optimisation terminée")
                
                # Visualisation comparative
                fig = go.Figure()
                
                # Trace Markowitz
                fig.add_trace(go.Scatter(
                    x=np.sqrt(risks_scal) * 100,
                    y=rets_scal * 100,
                    mode='lines+markers',
                    name='Markowitz (Benchmark)',
                    marker=dict(size=6, color='blue')
                ))
                
                # Trace NSGA-II
                rets_nsga = -F_nsga[:, 0]
                risks_nsga = F_nsga[:, 1]
                
                fig.add_trace(go.Scatter(
                    x=np.sqrt(risks_nsga) * 100,
                    y=rets_nsga * 100,
                    mode='markers',
                    name='NSGA-II (Approximation)',
                    marker=dict(size=6, color='red', symbol='x')
                ))
                
                fig.update_layout(
                    title="Superposition des Frontières Efficaces",
                    xaxis_title="Volatilité (%)",
                    yaxis_title="Rendement Espéré (%)",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Métriques simples
                metrics_df = compare_fronts([F_scal, F_nsga], ['Markowitz', 'NSGA-II'])
                st.subheader("Indicateurs de Convergence")
                st.dataframe(metrics_df[['name', 'hypervolume', 'spacing', 'spread']])

    # === 2. BOOTSTRAP (ROBUSTESSE DEMANDÉE) ===
    elif "Bootstrap" in analysis_type:
        st.header("🔄 Robustesse par Rééchantillonnage")
        
        st.markdown("""
        **Objectif PDF :** "Une procédure de rééchantillonnage permet d'évaluer la stabilité des portefeuilles."
        Nous utilisons le **Bootstrap** pour générer des scénarios de marché alternatifs et observer la dispersion des poids optimaux.
        """)
        
        n_bootstrap = st.slider("Nombre d'échantillons bootstrap", 20, 100, 50)
        
        if st.button("🔄 Lancer le Test de Stabilité"):
            with st.spinner(f"Génération de {n_bootstrap} scénarios..."):
                # 1. Génération des échantillons
                bootstrap_samples = bootstrap_resampling(returns, n_samples=n_bootstrap)
                
                # 2. Optimisation sur chaque échantillon (Portefeuille Tangent)
                w_mean, w_std = optimize_with_bootstrap(
                    mu, Sigma, bootstrap_samples,
                    optimizer_func=find_tangency_portfolio,
                    rf=0.02
                )
                
                st.success("✅ Analyse terminée")
                
                # Visualisation de l'instabilité
                st.subheader("Stabilité des Allocations (Top 10 Actifs)")
                
                # On trie par poids moyen
                top_indices = np.argsort(w_mean)[-10:][::-1]
                
                fig = go.Figure()
                
                for idx in top_indices:
                    ticker = tickers[idx]
                    weight = w_mean[idx] * 100
                    error = 1.96 * w_std[idx] * 100  # Intervalle de confiance 95%
                    
                    fig.add_trace(go.Bar(
                        x=[ticker], y=[weight],
                        name=ticker,
                        error_y=dict(type='data', array=[error], visible=True),
                        marker_color='steelblue'
                    ))
                
                fig.update_layout(
                    title="Poids Moyens et Incertitude (IC 95%)",
                    yaxis_title="Poids (%)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interprétation :** Les barres d'erreur indiquent la sensibilité de l'allocation aux données. 
                Une grande barre signifie que l'actif est sélectionné "par chance" dans certains scénarios, mais pas structurellement.
                """)

    # === 3. ANALYSE DES LIMITES (DEMANDÉE DANS LE RAPPORT) ===
    elif "Limites" in analysis_type:
        st.header("🔍 Analyse des Limites du Modèle")
        st.markdown("Analyse statistique justifiant les limites discutées dans le rapport.")
        
        tabs = st.tabs(["📉 Non-Normalité", "📊 Stationnarité", "❌ Erreurs d'Estimation"])
        
        with tabs[0]:
            st.subheader("Test de Normalité (Jarque-Bera)")
            if st.button("Lancer le test de normalité"):
                res = test_normality(returns)
                n_reject = res['JB_reject'].sum()
                st.warning(f"⚠️ {n_reject} actifs sur {len(tickers)} rejettent l'hypothèse de normalité.")
                st.markdown("Cela confirme que la variance (Markowitz) sous-estime les risques extrêmes.")
                st.dataframe(res.head(10))

        with tabs[1]:
            st.subheader("Test de Stationnarité")
            if st.button("Vérifier la stationnarité"):
                st.markdown("Graphique des statistiques glissantes sur un actif représentatif :")
                fig = plot_rolling_statistics(returns, tickers[0], window=252)
                st.pyplot(fig)
                st.markdown("La moyenne et la variance changent dans le temps, violant les hypothèses du modèle.")

        with tabs[2]:
            st.subheader("Sensibilité aux Erreurs")
            if st.button("Simuler les erreurs d'estimation"):
                errors = estimation_error_analysis(returns, n_iterations=30)
                st.metric("Erreur Moyenne sur les Rendements", f"{errors['mu_error_mean']:.4f}")
                st.metric("Erreur Moyenne sur la Covariance", f"{errors['sigma_error_mean']:.4f}")
                st.info("Ces erreurs expliquent l'instabilité observée dans le Bootstrap.")

if __name__ == "__main__":
    page_niveau3()