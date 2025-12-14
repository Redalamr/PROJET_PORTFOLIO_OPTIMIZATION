"""
Application Streamlit - Niveau 3 : Robustesse et Analyse des Limites
VERSION STRICTE CONFORME AU PDF :
- Niveau 1 : Comparaison Méthodes
- Niveau 2 : Optimisation Tri-Critère (Rendement/Risque/Coûts) + Sélection Interactive
- Niveau 3 : Robustesse (Bootstrap)
- Analyse des Limites
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Ajoute le dossier src au path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import prepare_data
from pareto_metrics import compare_fronts
from robustness import bootstrap_resampling, optimize_with_bootstrap
from limits_analysis import (
    test_normality, plot_rolling_statistics, estimation_error_analysis, 
    concentration_analysis
)
from optimizers.classic import generate_efficient_frontier_scalarization, find_tangency_portfolio
from optimizers.genetic import (
    optimize_nsga2_biobjective, 
    optimize_nsga2,  # Optimiseur Tri-objectif
    extract_pareto_front, 
    select_portfolio_from_front
)

def page_niveau3():
    st.title("🚀 Application de Gestion de Portefeuille (Niveaux 1, 2 & 3)")
    st.markdown("---")
    
    # === CHARGEMENT DES DONNÉES ===
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
    
    # === MENU DE SÉLECTION ===
    analysis_type = st.sidebar.radio(
        "Choisir l'étape du projet :",
        [
            "1️⃣ Niveau 1 : Comparaison (Markowitz vs NSGA-II)",
            "2️⃣ Niveau 2 : Contraintes & Coûts (Tri-Critère)",
            "3️⃣ Niveau 3 : Robustesse (Bootstrap)",
            "🔍 Analyse des Limites"
        ]
    )
    
    # =================================================================================
    # NIVEAU 1 : COMPARAISON MARKOWITZ VS NSGA-II
    # =================================================================================
    if "Niveau 1" in analysis_type:
        st.header("📊 Niveau 1 : Frontière Efficace Classique")
        st.info("Comparaison du modèle mathématique exact (Scalarisation) et de l'heuristique (NSGA-II Bi-objectif).")
        
        col1, col2 = st.columns(2)
        with col1:
            n_points = st.slider("Nombre de points (Scalarisation)", 20, 100, 50)
        with col2:
            n_gen = st.slider("Générations NSGA-II", 50, 200, 100)
        
        if st.button("🚀 Lancer la Comparaison"):
            with st.spinner("Calcul en cours..."):
                # 1. Scalarisation (Benchmark)
                portfolios_scal, rets_scal, risks_scal = generate_efficient_frontier_scalarization(
                    mu, Sigma, n_points=n_points
                )
                F_scal = np.column_stack([-rets_scal, risks_scal])
                
                # 2. NSGA-II (Bi-objectif)
                res_nsga = optimize_nsga2_biobjective(mu, Sigma, K=None, pop_size=100, n_gen=n_gen)
                X_nsga, F_nsga = extract_pareto_front(res_nsga)
                
                # Visualisation
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.sqrt(risks_scal) * 100, y=rets_scal * 100,
                    mode='lines+markers', name='Markowitz (Exact)',
                    marker=dict(size=6, color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=np.sqrt(F_nsga[:, 1]) * 100, y=-F_nsga[:, 0] * 100,
                    mode='markers', name='NSGA-II (Approx)',
                    marker=dict(size=6, color='red', symbol='x')
                ))
                fig.update_layout(title="Comparaison des Frontières", xaxis_title="Volatilité (%)", yaxis_title="Rendement (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Métriques
                st.subheader("Indicateurs de Performance")
                metrics = compare_fronts([F_scal, F_nsga], ['Markowitz', 'NSGA-II'])
                st.dataframe(metrics[['name', 'hypervolume', 'spread']])

    # =================================================================================
    # NIVEAU 2 : TRI-OBJECTIF & SÉLECTION (AJOUT MAJEUR)
    # =================================================================================
    elif "Niveau 2" in analysis_type:
        st.header("⚖️ Niveau 2 : Optimisation Tri-Critère avec Coûts")
        st.markdown("""
        **Objectifs :** Maximiser le rendement, Minimiser le risque, Minimiser les coûts de transaction.
        **Contraintes :** Cardinalité (nombre d'actifs) et budget.
        """)
        
        # Paramètres utilisateur
        col1, col2, col3 = st.columns(3)
        with col1:
            K_target = st.number_input("Cardinalité Cible (K)", min_value=2, max_value=len(tickers), value=10)
        with col2:
            c_prop = st.number_input("Coût Transaction (%)", value=0.5, step=0.1) / 100.0
        with col3:
            n_gen_tri = st.number_input("Générations", value=100, min_value=50)

        # Bouton de lancement
        if st.button("⚡ Calculer le Front de Pareto 3D"):
            with st.spinner("Optimisation NSGA-II Tri-Objectif en cours..."):
                # Simulation d'un portefeuille courant (ex: équipondéré) pour calculer les coûts de réallocation
                w_current = np.ones(len(tickers)) / len(tickers)
                
                res_tri = optimize_nsga2(
                    mu, Sigma, w_current=w_current, 
                    K=K_target, c_prop=c_prop, 
                    pop_size=100, n_gen=n_gen_tri
                )
                
                # Sauvegarde en session pour interactivité sans tout recalculer
                X_tri, F_tri = extract_pareto_front(res_tri)
                st.session_state['X_tri'] = X_tri
                st.session_state['F_tri'] = F_tri
                st.success(f"✅ Optimisation terminée : {len(F_tri)} solutions trouvées.")

        # Si résultats disponibles, on affiche l'interface de sélection
        if 'F_tri' in st.session_state:
            F = st.session_state['F_tri']
            X = st.session_state['X_tri']
            
            # 1. Graphique 3D
            st.subheader("1. Visualisation du Front de Pareto 3D")
            
            # Données pour le plot : Rendement (positif), Risque, Coûts
            returns_pct = -F[:, 0] * 100
            risks_pct = np.sqrt(F[:, 1]) * 100
            costs_pct = F[:, 2] * 100
            
            fig_3d = px.scatter_3d(
                x=risks_pct, y=returns_pct, z=costs_pct,
                color=costs_pct,
                labels={'x': 'Risque (%)', 'y': 'Rendement (%)', 'z': 'Coûts (%)'},
                title="Frontière Tri-Critère"
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 2. Outil de Sélection Interactive (Livrable PDF)
            st.subheader("2. Sélection du Portefeuille Optimal")
            st.markdown("Choisissez un **rendement minimal ($r_{min}$)**. L'outil sélectionnera le meilleur compromis Risque/Coûts.")
            
            min_r_slider = st.slider(
                "Rendement Minimal Souhaité (%)", 
                min_value=float(returns_pct.min()), 
                max_value=float(returns_pct.max()), 
                value=float(returns_pct.mean())
            )
            
            try:
                # Sélection du meilleur compromis respectant la contrainte
                # Note : On passe le rendement brut (non %) à la fonction
                idx, w_opt = select_portfolio_from_front(
                    X, F, min_return=(min_r_slider / 100.0)
                )
                
                # Affichage des métriques du portefeuille choisi
                st.write("### 🏆 Portefeuille Sélectionné")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Rendement Espéré", f"{returns_pct[idx]:.2f} %")
                col_m2.metric("Risque (Volatilité)", f"{risks_pct[idx]:.2f} %")
                col_m3.metric("Coûts Transaction", f"{costs_pct[idx]:.2f} %")
                
                # 3. Analyse Sectorielle (Demande PDF : "Ventilation par types d'industrie")
                st.subheader("3. Structure Macro-économique (Secteurs)")
                
                conc_analysis = concentration_analysis(w_opt, tickers, ticker_sectors)
                sector_weights = conc_analysis['sector_weights']
                
                # Camembert des secteurs
                df_sectors = pd.DataFrame(list(sector_weights.items()), columns=['Secteur', 'Poids'])
                df_sectors = df_sectors[df_sectors['Poids'] > 0.01] # Filtrer les tout petits
                
                fig_pie = px.pie(
                    df_sectors, values='Poids', names='Secteur',
                    title="Allocation Sectorielle du Portefeuille",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Top Actifs
                st.subheader("🔍 Top 10 Actifs")
                df_assets = pd.DataFrame({'Ticker': tickers, 'Poids': w_opt * 100})
                df_assets = df_assets.sort_values(by='Poids', ascending=False).head(10)
                st.bar_chart(df_assets.set_index('Ticker'))
                
            except ValueError as e:
                st.error(f"Aucun portefeuille ne respecte ce critère : {e}")

    # =================================================================================
    # NIVEAU 3 : ROBUSTESSE (BOOTSTRAP)
    # =================================================================================
    elif "Niveau 3" in analysis_type:
        st.header("🔄 Niveau 3 : Robustesse par Rééchantillonnage")
        st.markdown("**Objectif :** Évaluer la stabilité des poids optimaux face à l'incertitude des données.")
        
        n_bootstrap = st.slider("Nombre d'échantillons bootstrap", 20, 100, 30)
        
        if st.button("🔄 Lancer le Test de Stabilité"):
            with st.spinner("Simulation Bootstrap en cours..."):
                bootstrap_samples = bootstrap_resampling(returns, n_samples=n_bootstrap)
                w_mean, w_std = optimize_with_bootstrap(
                    mu, Sigma, bootstrap_samples,
                    optimizer_func=find_tangency_portfolio, rf=0.02
                )
                
                # Visualisation Top 15 Stabilité
                top_indices = np.argsort(w_mean)[-15:][::-1]
                fig = go.Figure()
                
                for idx in top_indices:
                    ticker = tickers[idx]
                    fig.add_trace(go.Bar(
                        x=[ticker], y=[w_mean[idx]*100],
                        name=ticker,
                        error_y=dict(type='data', array=[1.96*w_std[idx]*100], visible=True),
                        marker_color='steelblue'
                    ))
                
                fig.update_layout(title="Poids Moyens et Incertitude (IC 95%)", yaxis_title="Poids (%)")
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================================
    # ANALYSE DES LIMITES
    # =================================================================================
    elif "Limites" in analysis_type:
        st.header("🔍 Analyse des Limites du Modèle")
        
        tabs = st.tabs(["📉 Non-Normalité", "📊 Stationnarité", "❌ Erreurs d'Estimation"])
        
        with tabs[0]:
            st.subheader("Test de Normalité (Jarque-Bera)")
            if st.button("Lancer Test Normalité"):
                res = test_normality(returns)
                st.write(f"⚠️ {res['JB_reject'].sum()} actifs rejettent l'hypothèse de normalité.")
                st.dataframe(res.head())

        with tabs[1]:
            st.subheader("Stationnarité (Moyenne/Variance glissante)")
            asset = st.selectbox("Choisir un actif", tickers)
            st.pyplot(plot_rolling_statistics(returns, asset))

        with tabs[2]:
            st.subheader("Sensibilité aux Erreurs d'Estimation")
            if st.button("Simuler Erreurs"):
                errors = estimation_error_analysis(returns)
                st.metric("Erreur Moyenne Covariance", f"{errors['sigma_error_mean']:.4f}")
                st.info("Montre l'écart entre paramètres estimés sur échantillon vs réalité.")

if __name__ == "__main__":
    st.set_page_config(page_title="Projet Portfolio - Niveau 3", layout="wide")
    page_niveau3()