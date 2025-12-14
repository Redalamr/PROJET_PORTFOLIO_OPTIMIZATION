"""
Application Streamlit pour l'optimisation de portefeuille multi-critère
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Ajoute le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import prepare_data, load_tickers_by_sector
from financial_metrics import (
    portfolio_return, portfolio_risk, portfolio_volatility,
    sharpe_ratio, sector_allocation, portfolio_cardinality
)
from optimizers.classic import (
    generate_efficient_frontier_scalarization,
    generate_efficient_frontier_epsilon,
    find_tangency_portfolio
)
from optimizers.genetic import (
    optimize_nsga2, optimize_nsga2_biobjective,
    extract_pareto_front, select_portfolio_from_front
)

# Configuration de la page
st.set_page_config(
    page_title="Optimisation de Portefeuille",
    page_icon="📈",
    layout="wide"
)

# Titre principal
st.title("📈 Optimisation de Portefeuille Multi-Critère")
st.markdown("---")

# Sidebar pour les paramètres
st.sidebar.header("⚙️ Paramètres")

# Chargement des données
@st.cache_data
def load_data():
    try:
        prices, returns, mu, Sigma, ticker_sectors = prepare_data(
            start_date="2020-01-01",
            end_date="2024-12-31",
            data_dir="data/raw"
        )
        tickers = list(prices.columns)
        return prices, returns, mu, Sigma, tickers, ticker_sectors
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None, None, None, None, None

# Charge les données
with st.spinner("Chargement des données..."):
    prices, returns, mu, Sigma, tickers, ticker_sectors = load_data()

if prices is None:
    st.stop()

st.sidebar.success(f"✅ {len(tickers)} actifs chargés")

# Sélection du niveau
niveau = st.sidebar.radio(
    "Niveau d'optimisation",
    ["Niveau 1: Markowitz Classique", "Niveau 2: Avec Contraintes", "Comparaison"],
    index=0
)

# Paramètres communs
st.sidebar.subheader("Paramètres Financiers")
rf = st.sidebar.slider("Taux sans risque (%)", 0.0, 10.0, 2.0, 0.5) / 100

if "Niveau 2" in niveau or "Comparaison" in niveau:
    st.sidebar.subheader("Contraintes Opérationnelles")
    K = st.sidebar.slider("Cardinalité (nombre d'actifs)", 5, 50, 20, 1)
    c_prop = st.sidebar.slider("Coût de transaction (%)", 0.0, 2.0, 0.5, 0.1) / 100
    
    # Portefeuille actuel (pour les coûts de transaction)
    use_current = st.sidebar.checkbox("Définir un portefeuille actuel", value=False)
    if use_current:
        w_current = np.ones(len(tickers)) / len(tickers)  # Équipondéré par défaut
    else:
        w_current = np.zeros(len(tickers))
else:
    K = None
    c_prop = 0.005
    w_current = np.zeros(len(tickers))

# === NIVEAU 1: MARKOWITZ CLASSIQUE ===
if "Niveau 1" in niveau:
    st.header("Niveau 1: Modèle de Markowitz Classique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Méthode de Résolution")
        method = st.radio(
            "Choisir une méthode",
            ["Scalarisation", "Epsilon-Contrainte"]
        )
    
    with col2:
        st.subheader("Paramètres")
        n_points = st.slider("Nombre de points sur la frontière", 20, 200, 50, 10)
    
    if st.button("🚀 Générer la Frontière Efficace", key="btn_niveau1"):
        with st.spinner("Optimisation en cours..."):
            if method == "Scalarisation":
                portfolios, rets, risks = generate_efficient_frontier_scalarization(
                    mu, Sigma, n_points=n_points
                )
            else:
                portfolios, rets, risks = generate_efficient_frontier_epsilon(
                    mu, Sigma, n_points=n_points
                )
            
            # Calcule les volatilités
            vols = np.sqrt(risks)
            
            # Trouve le portefeuille tangent
            w_tangent = find_tangency_portfolio(mu, Sigma, rf)
            ret_tangent = portfolio_return(w_tangent, mu)
            vol_tangent = portfolio_volatility(w_tangent, Sigma)
            
            # Graphique de la frontière
            fig = go.Figure()
            
            # Frontière efficace
            fig.add_trace(go.Scatter(
                x=vols * 100,
                y=rets * 100,
                mode='markers',
                name='Frontière Efficace',
                marker=dict(size=6, color=rets, colorscale='Viridis', showscale=True),
                hovertemplate='<b>Risque:</b> %{x:.2f}%<br><b>Rendement:</b> %{y:.2f}%<extra></extra>'
            ))
            
            # Portefeuille tangent
            fig.add_trace(go.Scatter(
                x=[vol_tangent * 100],
                y=[ret_tangent * 100],
                mode='markers',
                name='Portefeuille Tangent',
                marker=dict(size=15, color='red', symbol='star')
            ))
            
            # Actifs individuels
            asset_vols = np.sqrt(np.diag(Sigma))
            fig.add_trace(go.Scatter(
                x=asset_vols * 100,
                y=mu * 100,
                mode='markers',
                name='Actifs Individuels',
                marker=dict(size=4, color='gray', opacity=0.5),
                hovertemplate='<b>Risque:</b> %{x:.2f}%<br><b>Rendement:</b> %{y:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Frontière Efficace - Méthode: {method}",
                xaxis_title="Volatilité (%)",
                yaxis_title="Rendement Annuel (%)",
                hovermode='closest',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques du portefeuille tangent
            st.subheader("📊 Portefeuille Tangent (Sharpe Maximum)")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rendement", f"{ret_tangent*100:.2f}%")
            col2.metric("Volatilité", f"{vol_tangent*100:.2f}%")
            sharpe_t = sharpe_ratio(w_tangent, mu, Sigma, rf)
            col3.metric("Sharpe Ratio", f"{sharpe_t:.3f}")
            card_t = portfolio_cardinality(w_tangent)
            col4.metric("Cardinalité", f"{card_t}")
            
            # Composition du portefeuille tangent
            st.subheader("Composition du Portefeuille Tangent")
            
            # Top 10 positions
            top_indices = np.argsort(w_tangent)[-10:][::-1]
            top_tickers = [tickers[i] for i in top_indices]
            top_weights = [w_tangent[i] for i in top_indices]
            
            df_comp = pd.DataFrame({
                'Ticker': top_tickers,
                'Poids (%)': [w*100 for w in top_weights],
                'Secteur': [ticker_sectors.get(t, 'Unknown') for t in top_tickers]
            })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df_comp, use_container_width=True)
            
            with col2:
                # Répartition sectorielle
                sector_weights = sector_allocation(w_tangent, tickers, ticker_sectors)
                df_sectors = pd.DataFrame(list(sector_weights.items()), 
                                         columns=['Secteur', 'Poids'])
                df_sectors['Poids (%)'] = df_sectors['Poids'] * 100
                
                fig_pie = px.pie(df_sectors, values='Poids (%)', names='Secteur',
                               title='Répartition Sectorielle')
                st.plotly_chart(fig_pie, use_container_width=True)

# === NIVEAU 2: AVEC CONTRAINTES ===
elif "Niveau 2" in niveau:
    st.header("Niveau 2: Optimisation avec Contraintes Opérationnelles")
    
    st.info(f"🎯 Contraintes: Cardinalité = {K} actifs, Coût de transaction = {c_prop*100:.2f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pop_size = st.slider("Taille de la population", 50, 300, 100, 50)
    
    with col2:
        n_gen = st.slider("Nombre de générations", 50, 300, 100, 50)
    
    if st.button("🚀 Lancer NSGA-II", key="btn_niveau2"):
        with st.spinner("Optimisation génétique en cours... (peut prendre quelques minutes)"):
            # Optimisation tri-objectif
            res = optimize_nsga2(
                mu=mu,
                Sigma=Sigma,
                w_current=w_current,
                K=K,
                c_prop=c_prop,
                pop_size=pop_size,
                n_gen=n_gen
            )
            
            X, F = extract_pareto_front(res)
            
            # F[:,0] = -rendement, F[:,1] = risque, F[:,2] = coûts
            returns_pf = -F[:, 0]
            risks_pf = F[:, 1]
            costs_pf = F[:, 2]
            vols_pf = np.sqrt(risks_pf)
            
            st.success(f"✅ Front de Pareto généré: {len(X)} solutions")
            
            # Visualisation 3D
            st.subheader("Front de Pareto 3D")
            
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=vols_pf * 100,
                y=returns_pf * 100,
                z=costs_pf * 100,
                mode='markers',
                marker=dict(
                    size=5,
                    color=returns_pf,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Rendement (%)")
                ),
                hovertemplate='<b>Volatilité:</b> %{x:.2f}%<br>' +
                             '<b>Rendement:</b> %{y:.2f}%<br>' +
                             '<b>Coûts:</b> %{z:.4f}%<extra></extra>'
            )])
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='Volatilité (%)',
                    yaxis_title='Rendement (%)',
                    zaxis_title='Coûts Transaction (%)'
                ),
                title="Front de Pareto Tri-Objectif",
                height=700
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Projections 2D
            st.subheader("Projections 2D du Front de Pareto")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=vols_pf * 100,
                    y=returns_pf * 100,
                    mode='markers',
                    marker=dict(size=5, color=costs_pf, colorscale='Reds', showscale=True)
                ))
                fig1.update_layout(
                    title="Rendement vs Risque",
                    xaxis_title="Volatilité (%)",
                    yaxis_title="Rendement (%)",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=vols_pf * 100,
                    y=costs_pf * 100,
                    mode='markers',
                    marker=dict(size=5, color=returns_pf, colorscale='Viridis')
                ))
                fig2.update_layout(
                    title="Coûts vs Risque",
                    xaxis_title="Volatilité (%)",
                    yaxis_title="Coûts (%)",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col3:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=returns_pf * 100,
                    y=costs_pf * 100,
                    mode='markers',
                    marker=dict(size=5, color=vols_pf, colorscale='Blues')
                ))
                fig3.update_layout(
                    title="Coûts vs Rendement",
                    xaxis_title="Rendement (%)",
                    yaxis_title="Coûts (%)",
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # Sélection d'un portefeuille
            st.subheader("🎯 Sélection d'un Portefeuille")
            
            min_return = st.slider(
                "Rendement minimum requis (%)",
                float(returns_pf.min() * 100),
                float(returns_pf.max() * 100),
                float(returns_pf.mean() * 100),
                0.1
            ) / 100
            
            try:
                idx, w_selected = select_portfolio_from_front(X, F, min_return=min_return)
                
                ret_sel = returns_pf[idx]
                risk_sel = risks_pf[idx]
                vol_sel = vols_pf[idx]
                cost_sel = costs_pf[idx]
                sharpe_sel = (ret_sel - rf) / vol_sel
                card_sel = portfolio_cardinality(w_selected)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Rendement", f"{ret_sel*100:.2f}%")
                col2.metric("Volatilité", f"{vol_sel*100:.2f}%")
                col3.metric("Coûts", f"{cost_sel*100:.3f}%")
                col4.metric("Sharpe", f"{sharpe_sel:.3f}")
                col5.metric("Cardinalité", f"{card_sel}/{K}")
                
                # Composition
                st.subheader("Composition du Portefeuille Sélectionné")
                
                # Filtre les positions non nulles
                active_positions = w_selected > 1e-4
                active_tickers = [tickers[i] for i in range(len(tickers)) if active_positions[i]]
                active_weights = [w_selected[i] for i in range(len(tickers)) if active_positions[i]]
                
                # Trie par poids décroissant
                sorted_indices = np.argsort(active_weights)[::-1]
                active_tickers = [active_tickers[i] for i in sorted_indices]
                active_weights = [active_weights[i] for i in sorted_indices]
                
                df_portfolio = pd.DataFrame({
                    'Ticker': active_tickers,
                    'Poids (%)': [w*100 for w in active_weights],
                    'Secteur': [ticker_sectors.get(t, 'Unknown') for t in active_tickers]
                })
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(df_portfolio, use_container_width=True, height=400)
                
                with col2:
                    sector_weights = sector_allocation(w_selected, tickers, ticker_sectors)
                    df_sectors = pd.DataFrame(list(sector_weights.items()),
                                             columns=['Secteur', 'Poids'])
                    df_sectors['Poids (%)'] = df_sectors['Poids'] * 100
                    df_sectors = df_sectors[df_sectors['Poids (%)'] > 0.1]  # Filtre les très petits
                    
                    fig_pie = px.pie(df_sectors, values='Poids (%)', names='Secteur',
                                   title='Répartition Sectorielle')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
            except ValueError as e:
                st.error(f"Aucun portefeuille ne satisfait les critères: {e}")

# === COMPARAISON ===
else:
    st.header("Comparaison des Méthodes")
    
    st.info("Cette section compare les résultats de Markowitz classique et NSGA-II")
    
    if st.button("🚀 Lancer la Comparaison", key="btn_compare"):
        with st.spinner("Optimisation en cours..."):
            # Markowitz
            portfolios_mark, rets_mark, risks_mark = generate_efficient_frontier_scalarization(
                mu, Sigma, n_points=50
            )
            vols_mark = np.sqrt(risks_mark)
            
            # NSGA-II bi-objectif
            res_nsga = optimize_nsga2_biobjective(mu, Sigma, K=None, pop_size=100, n_gen=100)
            X_nsga, F_nsga = extract_pareto_front(res_nsga)
            rets_nsga = -F_nsga[:, 0]
            risks_nsga = F_nsga[:, 1]
            vols_nsga = np.sqrt(risks_nsga)
            
            # Graphique comparatif
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=vols_mark * 100,
                y=rets_mark * 100,
                mode='markers',
                name='Markowitz (Scalarisation)',
                marker=dict(size=8, color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=vols_nsga * 100,
                y=rets_nsga * 100,
                mode='markers',
                name='NSGA-II',
                marker=dict(size=8, color='red')
            ))
            
            fig.update_layout(
                title="Comparaison: Markowitz vs NSGA-II",
                xaxis_title="Volatilité (%)",
                yaxis_title="Rendement (%)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques comparatives
            st.subheader("Statistiques Comparatives")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Markowitz (Scalarisation)**")
                st.write(f"- Nombre de solutions: {len(rets_mark)}")
                st.write(f"- Rendement min: {rets_mark.min()*100:.2f}%")
                st.write(f"- Rendement max: {rets_mark.max()*100:.2f}%")
                st.write(f"- Risque min: {vols_mark.min()*100:.2f}%")
                st.write(f"- Risque max: {vols_mark.max()*100:.2f}%")
            
            with col2:
                st.markdown("**NSGA-II**")
                st.write(f"- Nombre de solutions: {len(rets_nsga)}")
                st.write(f"- Rendement min: {rets_nsga.min()*100:.2f}%")
                st.write(f"- Rendement max: {rets_nsga.max()*100:.2f}%")
                st.write(f"- Risque min: {vols_nsga.min()*100:.2f}%")
                st.write(f"- Risque max: {vols_nsga.max()*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("**Projet d'Optimisation de Portefeuille Multi-Critère** | Décembre 2025")