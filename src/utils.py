"""
Fonctions utilitaires pour l'analyse et la visualisation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def plot_efficient_frontier(returns: np.ndarray, risks: np.ndarray, 
                           method_name: str = "Efficient Frontier",
                           save_path: str = None):
    """
    Crée un graphique de la frontière efficace.
    
    Args:
        returns: Rendements des portefeuilles
        risks: Risques (variance) des portefeuilles
        method_name: Nom de la méthode utilisée
        save_path: Chemin pour sauvegarder le graphique
    """
    vols = np.sqrt(risks)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(vols * 100, returns * 100, c=returns, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(label='Rendement (%)')
    plt.xlabel('Volatilité (%)')
    plt.ylabel('Rendement Annuel (%)')
    plt.title(f'Frontière Efficace - {method_name}')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return plt.gcf()

def plot_correlation_matrix(returns: pd.DataFrame, title: str = "Matrice de Corrélation",
                           figsize: Tuple[int, int] = (12, 10)):
    """
    Visualise la matrice de corrélation des rendements.
    
    Args:
        returns: DataFrame des rendements
        title: Titre du graphique
        figsize: Taille de la figure
    """
    corr = returns.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_portfolio_composition(weights: np.ndarray, tickers: List[str], 
                              top_n: int = 10, title: str = "Composition du Portefeuille"):
    """
    Visualise la composition d'un portefeuille (top N positions).
    
    Args:
        weights: Vecteur des poids
        tickers: Liste des noms de tickers
        top_n: Nombre de positions à afficher
        title: Titre du graphique
    """
    # Sélectionne les top N positions
    top_indices = np.argsort(weights)[-top_n:][::-1]
    top_tickers = [tickers[i] for i in top_indices]
    top_weights = [weights[i] * 100 for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_tickers, top_weights, color='steelblue')
    plt.xlabel('Poids (%)')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_sector_allocation(weights: np.ndarray, tickers: List[str], 
                          ticker_sectors: dict, title: str = "Répartition Sectorielle"):
    """
    Visualise la répartition sectorielle d'un portefeuille.
    
    Args:
        weights: Vecteur des poids
        tickers: Liste des noms de tickers
        ticker_sectors: Dictionnaire {ticker: secteur}
        title: Titre du graphique
    """
    from financial_metrics import sector_allocation
    
    sector_weights = sector_allocation(weights, tickers, ticker_sectors)
    
    # Trie par poids décroissant
    sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
    sectors = [s[0] for s in sorted_sectors]
    weights_pct = [s[1] * 100 for s in sorted_sectors]
    
    plt.figure(figsize=(10, 6))
    plt.pie(weights_pct, labels=sectors, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    return plt.gcf()

def plot_pareto_front_3d(objectives: np.ndarray, labels: List[str] = None):
    """
    Visualise un front de Pareto 3D.
    
    Args:
        objectives: Matrice (n_solutions, 3) des objectifs
        labels: Labels des axes [x, y, z]
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    if labels is None:
        labels = ['Objectif 1', 'Objectif 2', 'Objectif 3']
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                        c=objectives[:, 0], cmap='viridis', s=50, alpha=0.6)
    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title('Front de Pareto 3D')
    
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    return fig

def calculate_summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des statistiques descriptives sur les rendements.
    
    Args:
        returns: DataFrame des rendements
        
    Returns:
        DataFrame avec les statistiques par actif
    """
    stats = pd.DataFrame({
        'Rendement Moyen': returns.mean() * 252,
        'Volatilité': returns.std() * np.sqrt(252),
        'Sharpe (rf=2%)': (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252)),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Min': returns.min(),
        'Max': returns.max()
    })
    
    return stats

def backtest_portfolio(weights: np.ndarray, returns: pd.DataFrame, 
                      initial_value: float = 100000) -> pd.DataFrame:
    """
    Simule la performance historique d'un portefeuille.
    
    Args:
        weights: Vecteur des poids
        returns: DataFrame des rendements historiques
        initial_value: Valeur initiale du portefeuille
        
    Returns:
        DataFrame avec la valeur du portefeuille au fil du temps
    """
    # Rendements du portefeuille
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Valeur cumulée
    cumulative_returns = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_value * cumulative_returns
    
    df_backtest = pd.DataFrame({
        'Date': returns.index,
        'Valeur': portfolio_value,
        'Rendement': portfolio_returns,
        'Rendement Cumulé': cumulative_returns - 1
    })
    
    return df_backtest

def plot_backtest(backtest_df: pd.DataFrame, title: str = "Performance Historique"):
    """
    Visualise les résultats d'un backtest.
    
    Args:
        backtest_df: DataFrame retourné par backtest_portfolio
        title: Titre du graphique
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Valeur du portefeuille
    ax1.plot(backtest_df['Date'], backtest_df['Valeur'], linewidth=2)
    ax1.set_ylabel('Valeur du Portefeuille')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    # Rendements quotidiens
    ax2.plot(backtest_df['Date'], backtest_df['Rendement'] * 100, linewidth=1, alpha=0.7)
    ax2.set_ylabel('Rendement Quotidien (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    return fig

def calculate_portfolio_metrics(weights: np.ndarray, mu: np.ndarray, 
                                Sigma: np.ndarray, rf: float = 0.02) -> dict:
    """
    Calcule toutes les métriques d'un portefeuille.
    
    Args:
        weights: Vecteur des poids
        mu: Rendements moyens
        Sigma: Matrice de covariance
        rf: Taux sans risque
        
    Returns:
        Dictionnaire des métriques
    """
    from financial_metrics import (
        portfolio_return, portfolio_risk, portfolio_volatility,
        sharpe_ratio, portfolio_cardinality
    )
    
    ret = portfolio_return(weights, mu)
    risk = portfolio_risk(weights, Sigma)
    vol = portfolio_volatility(weights, Sigma)
    sharpe = sharpe_ratio(weights, mu, Sigma, rf)
    card = portfolio_cardinality(weights)
    
    return {
        'Rendement Annuel (%)': ret * 100,
        'Risque (Variance)': risk,
        'Volatilité (%)': vol * 100,
        'Sharpe Ratio': sharpe,
        'Cardinalité': card,
        'Poids Max (%)': weights.max() * 100,
        'Poids Min (%)': weights[weights > 0].min() * 100 if np.any(weights > 0) else 0
    }

def format_portfolio_table(weights: np.ndarray, tickers: List[str],
                          mu: np.ndarray, threshold: float = 0.001) -> pd.DataFrame:
    """
    Crée un tableau formaté de la composition du portefeuille.
    
    Args:
        weights: Vecteur des poids
        tickers: Liste des tickers
        mu: Rendements moyens
        threshold: Seuil en-dessous duquel les positions sont ignorées
        
    Returns:
        DataFrame formaté
    """
    # Filtre les positions significatives
    mask = weights > threshold
    
    df = pd.DataFrame({
        'Ticker': [tickers[i] for i in range(len(tickers)) if mask[i]],
        'Poids (%)': [weights[i] * 100 for i in range(len(tickers)) if mask[i]],
        'Rendement Moyen (%)': [mu[i] * 100 for i in range(len(tickers)) if mask[i]]
    })
    
    # Trie par poids décroissant
    df = df.sort_values('Poids (%)', ascending=False).reset_index(drop=True)
    
    return df

def generate_latex_table(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """
    Génère le code LaTeX pour un tableau de résultats.
    
    Args:
        df: DataFrame à convertir
        caption: Légende du tableau
        label: Label pour référence
        
    Returns:
        String contenant le code LaTeX
    """
    latex_str = df.to_latex(index=False, float_format="%.3f")
    
    if caption:
        latex_str = latex_str.replace("\\begin{tabular}", 
                                     f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}")
    
    return latex_str

def compare_portfolios(weights_list: List[np.ndarray], 
                      names: List[str],
                      mu: np.ndarray,
                      Sigma: np.ndarray,
                      rf: float = 0.02) -> pd.DataFrame:
    """
    Compare plusieurs portefeuilles côte à côte.
    
    Args:
        weights_list: Liste de vecteurs de poids
        names: Noms des portefeuilles
        mu: Rendements moyens
        Sigma: Matrice de covariance
        rf: Taux sans risque
        
    Returns:
        DataFrame comparatif
    """
    results = []
    
    for weights, name in zip(weights_list, names):
        metrics = calculate_portfolio_metrics(weights, mu, Sigma, rf)
        metrics['Nom'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[['Nom'] + [col for col in df.columns if col != 'Nom']]
    
    return df