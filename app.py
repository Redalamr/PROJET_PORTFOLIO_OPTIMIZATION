"""
Application Streamlit pour l'optimisation de portefeuille multi-critère
Conforme aux exigences du projet final (Niveaux 1, 2 et 3)
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

# --- IMPORTS CORRIGÉS ---
from data_loader import prepare_data
from financial_metrics import (
    portfolio_return, portfolio_volatility,
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
from pareto_metrics import compare_fronts
from robustness import (
    bootstrap_resampling, optimize_with_bootstrap
)
from limits_analysis import (
    test_normality, test_stationarity, 
    estimation_error_analysis
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
    [
        "Niveau 1: Markowitz Classique",
        "Niveau 2: Avec Contraintes & Coûts",
        "Niveau 3: Robustesse & Limites"
    ],
    index=0
)

# Paramètres communs
st.sidebar.subheader("Paramètres Financiers")
rf = st.sidebar.slider("Taux sans risque (%)", 0.0, 10.0, 2.0, 0.5) / 100

# Paramètres spécifiques au Niveau 2
if "Niveau 2" in niveau:
    st.sidebar.subheader("Contraintes Opérationnelles")
    K = st.sidebar.slider("Cardinalité (nombre d'actifs)", 5, 50, 20, 1)
    c_prop = st.sidebar.slider("Coût de transaction (%)", 0.0, 2.0, 0.5, 0.1) / 100
    
    use_current = st.sidebar.checkbox("Définir un portefeuille actuel", value=False)
    if use_current:
        w_current = np.ones(len(tickers)) / len(tickers)
    else:
        w_current = np.zeros(len(tickers))
else:
    # Valeurs par défaut pour les autres niveaux
    K = None
    c_prop = 0.005
    w_current = np.zeros(len(tickers))

# =================================================================================
# NIVEAU 1: MARKOWITZ CLASSIQUE
# =================================================================================
if "Niveau 1" in niveau:
    st.header("Niveau 1: Modèle de Markowitz Classique")
    st.markdown("**Objectif :** Optimiser le couple Rendement / Risque ($f_1, f_2$) sous contraintes de base.")
    
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
            
            vols = np.sqrt(risks)
            
            # Portefeuille Tangent
            w_tangent = find_tangency_portfolio(mu, Sigma, rf)
            ret_tangent = portfolio_return(w_tangent, mu)
            vol_tangent = portfolio_volatility(w_tangent, Sigma)
            
            # --- Graphique ---
            fig = go.Figure()
            
            # Frontière
            fig.add_trace(go.Scatter(
                x=vols * 100, y=rets * 100,
                mode='markers', name='Frontière Efficace',
                marker=dict(size=6, color=rets, colorscale='Viridis', showscale=True),
                hovertemplate='Risque: %{x:.2f}%<br>Rendement: %{y:.2f}%'
            ))
            
            # Tangent
            fig.add_trace(go.Scatter(
                x=[vol_tangent * 100], y=[ret_tangent * 100],
                mode='markers', name='Portefeuille Tangent',
                marker=dict(size=15, color='red', symbol='star')
            ))
            
            fig.update_layout(
                title=f"Frontière Efficace ({method})",
                xaxis_title="Volatilité (%)",
                yaxis_title="Rendement Annuel (%)",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Résultats ---
            st.subheader("📊 Portefeuille Tangent")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rendement", f"{ret_tangent*100:.2f}%")
            c2.metric("Volatilité", f"{vol_tangent*100:.2f}%")
            c3.metric("Sharpe Ratio", f"{sharpe_ratio(w_tangent, mu, Sigma, rf):.3f}")

            # Répartition Sectorielle
            st.subheader("Structure du Portefeuille")
            sector_weights = sector_allocation(w_tangent, tickers, ticker_sectors)
            df_sectors = pd.DataFrame(list(sector_weights.items()), columns=['Secteur', 'Poids'])
            df_sectors = df_sectors[df_sectors['Poids'] > 0.001] # Filtre affichage
            
            fig_pie = px.pie(df_sectors, values='Poids', names='Secteur', title='Répartition Sectorielle')
            st.plotly_chart(fig_pie, use_container_width=True)

# =================================================================================
# NIVEAU 2: CONTRAINTES & COÛTS
# =================================================================================
elif "Niveau 2" in niveau:
    st.header("Niveau 2: Optimisation avec Contraintes Opérationnelles")
    st.markdown("""
    **Objectif :** Intégrer la **Cardinalité** (nombre fixe d'actifs $K$) et les **Coûts de transaction** ($f_3$).
    Utilisation de l'algorithme génétique **NSGA-II**.
    """)
    
    st.info(f"Paramètres : Cardinalité K={K}, Coûts={c_prop*100:.2f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        pop_size = st.slider("Taille de la population", 50, 300, 100, 50)
    with col2:
        n_gen = st.slider("Nombre de générations", 20, 300, 50, 10) # Réduit par défaut pour rapidité
    
    if st.button("🚀 Lancer NSGA-II", key="btn_niveau2"):
        with st.spinner("Optimisation génétique en cours..."):
            # Optimisation Tri-Objectif (Rendement, Risque, Coûts)
            res = optimize_nsga2(
                mu=mu, Sigma=Sigma, w_current=w_current,
                K=K, c_prop=c_prop,
                pop_size=pop_size, n_gen=n_gen
            )
            
            X, F = extract_pareto_front(res)
            
            # F[:,0] = -Rendement, F[:,1] = Risque, F[:,2] = Coûts
            returns_pf = -F[:, 0]
            risks_pf = F[:, 1]
            costs_pf = F[:, 2]
            vols_pf = np.sqrt(risks_pf)
            
            st.success(f"✅ Front de Pareto généré : {len(X)} solutions")
            
            # --- Visualisation 3D ---
            st.subheader("Front de Pareto 3D")
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=vols_pf * 100,
                y=returns_pf * 100,
                z=costs_pf * 100,
                mode='markers',
                marker=dict(
                    size=5, color=returns_pf, colorscale='Viridis',
                    colorbar=dict(title="Rendement")
                ),
                hovertemplate='Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<br>Coût: %{z:.3f}%'
            )])
            fig_3d.update_layout(
                scene=dict(xaxis_title='Volatilité (%)', yaxis_title='Rendement (%)', zaxis_title='Coûts (%)'),
                height=600, margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # --- Sélection Interactives ---
            st.markdown("---")
            st.subheader("🎯 Sélection d'un Portefeuille")
            
            min_return_select = st.slider(
                "Rendement minimum souhaité (%)",
                float(returns_pf.min() * 100), float(returns_pf.max() * 100),
                float(returns_pf.mean() * 100)
            ) / 100
            
            try:
                # Sélection basée sur le meilleur compromis satisfaisant la contrainte
                idx, w_selected = select_portfolio_from_front(X, F, min_return=min_return_select)
                
                # Métriques du portefeuille choisi
                ret_sel = returns_pf[idx]
                vol_sel = vols_pf[idx]
                cost_sel = costs_pf[idx]
                card_sel = portfolio_cardinality(w_selected)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rendement", f"{ret_sel*100:.2f}%")
                c2.metric("Volatilité", f"{vol_sel*100:.2f}%")
                c3.metric("Coûts Trans.", f"{cost_sel*100:.3f}%")
                c4.metric("Cardinalité", f"{card_sel}/{K}")
                
                # Composition et Secteurs
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.caption("Top 10 Actifs")
                    # Filtrage et tri
                    mask = w_selected > 0.001
                    sel_tickers = [tickers[i] for i in range(len(tickers)) if mask[i]]
                    sel_weights = w_selected[mask]
                    # Tri décroissant
                    sort_idx = np.argsort(sel_weights)[::-1]
                    
                    df_p = pd.DataFrame({
                        'Ticker': [sel_tickers[i] for i in sort_idx][:10],
                        'Poids (%)': [sel_weights[i]*100 for i in sort_idx][:10]
                    })
                    st.dataframe(df_p, use_container_width=True)
                
                with col_right:
                    st.caption("Répartition Sectorielle")
                    sect_weights = sector_allocation(w_selected, tickers, ticker_sectors)
                    df_s = pd.DataFrame(list(sect_weights.items()), columns=['Secteur', 'Poids'])
                    df_s = df_s[df_s['Poids'] > 0.01]
                    fig_p = px.pie(df_s, values='Poids', names='Secteur')
                    fig_p.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_p, use_container_width=True)
                    
            except ValueError as e:
                st.warning(f"Aucun portefeuille ne respecte ce rendement minimal ({e})")

# =================================================================================
# NIVEAU 3: ROBUSTESSE ET LIMITES
# =================================================================================
elif "Niveau 3" in niveau:
    st.header("Niveau 3: Robustesse et Analyse des Limites")
    
    tab1, tab2, tab3 = st.tabs(["🔄 Robustesse (Bootstrap)", "📉 Limites du Modèle", "📊 Comparaison Méthodes"])
    
    # --- TAB 1: BOOTSTRAP ---
    with tab1:
        st.subheader("Stabilité des Portefeuilles (Bootstrap)")
        st.markdown("La procédure de rééchantillonnage permet d'évaluer la sensibilité des poids optimaux aux variations des données historiques.")
        
        n_boot = st.slider("Nombre de simulations", 20, 100, 30)
        
        if st.button("Lancer le Test de Stabilité"):
            with st.spinner("Rééchantillonnage et optimisation..."):
                # 1. Génération scénarios
                samples = bootstrap_resampling(returns, n_samples=n_boot)
                
                # 2. Optimisation sur chaque scénario (Portefeuille Tangent comme référence)
                w_mean, w_std = optimize_with_bootstrap(
                    mu, Sigma, samples,
                    optimizer_func=find_tangency_portfolio,
                    rf=rf
                )
                
                # Visualisation
                top_idx = np.argsort(w_mean)[-10:][::-1]
                
                fig = go.Figure()
                for i in top_idx:
                    ticker = tickers[i]
                    # Moyenne
                    fig.add_trace(go.Bar(
                        x=[ticker], y=[w_mean[i]*100],
                        name=ticker,
                        marker_color='steelblue',
                        error_y=dict(type='data', array=[1.96*w_std[i]*100], visible=True) # IC 95%
                    ))
                
                fig.update_layout(
                    title="Poids Moyens et Incertitude (Intervalle de Confiance 95%)",
                    yaxis_title="Poids (%)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("Les barres d'erreur montrent l'instabilité : une barre large indique que l'allocation à cet actif dépend fortement de l'échantillon de temps spécifique.")

    # --- TAB 2: LIMITES ---
    with tab2:
        st.subheader("Limites Statistiques du Modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 1. Non-Normalité")
            if st.button("Test de Normalité"):
                res_norm = test_normality(returns)
                n_reject = res_norm['JB_reject'].sum()
                st.warning(f"⚠️ {n_reject} actifs sur {len(tickers)} ne suivent pas une loi Normale.")
                st.dataframe(res_norm.head())
        
        with col2:
            st.markdown("### 2. Erreur d'Estimation")
            if st.button("Simuler Erreurs"):
                err = estimation_error_analysis(returns, n_iterations=20)
                st.metric("Erreur Moyenne (Rendements)", f"{err['mu_error_mean']:.2%}")
                st.metric("Erreur Moyenne (Covariance)", f"{err['sigma_error_mean']:.2%}")
                st.caption("Différence moyenne entre les paramètres estimés sur un sous-échantillon et la réalité.")

    # --- TAB 3: COMPARAISON ---
    with tab3:
        st.subheader("Markowitz (Convexe) vs NSGA-II (Heuristique)")
        
        if st.button("Comparer les Frontières"):
            with st.spinner("Calcul..."):
                # Markowitz
                _, rets_m, risks_m = generate_efficient_frontier_scalarization(mu, Sigma, n_points=30)
                F_mark = np.column_stack([-rets_m, risks_m])
                
                # NSGA-II (Biobjectif pour comparer équitablement)
                res_nsga = optimize_nsga2_biobjective(mu, Sigma, K=None, pop_size=50, n_gen=50)
                _, F_nsga = extract_pareto_front(res_nsga)
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.sqrt(risks_m)*100, y=rets_m*100, mode='lines', name='Markowitz (Exact)'))
                fig.add_trace(go.Scatter(x=np.sqrt(F_nsga[:,1])*100, y=-F_nsga[:,0]*100, mode='markers', name='NSGA-II (Approx)'))
                
                fig.update_layout(title="Comparaison des Frontières", xaxis_title="Volatilité (%)", yaxis_title="Rendement (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Métriques
                df_comp = compare_fronts([F_mark, F_nsga], ['Markowitz', 'NSGA-II'])
                st.dataframe(df_comp[['name', 'hypervolume', 'spacing']])

# Footer
st.markdown("---")
st.markdown("**Projet Optimisation de Portefeuille** | Décembre 2025")