"""
Analyse des limites du modèle de Markowitz
À placer dans : src/limits_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, List, Dict

def test_normality(returns: pd.DataFrame, 
                   alpha: float = 0.05) -> pd.DataFrame:
    """
    Teste la normalité des rendements avec plusieurs tests statistiques.
    
    Args:
        returns: DataFrame des rendements
        alpha: Seuil de significativité
        
    Returns:
        DataFrame avec les résultats des tests
    """
    results = []
    
    for col in returns.columns:
        data = returns[col].dropna()
        
        # Test de Jarque-Bera
        jb_stat, jb_pvalue = stats.jarque_bera(data)
        
        # Test de Shapiro-Wilk (sur échantillon si trop grand)
        if len(data) > 5000:
            sw_stat, sw_pvalue = stats.shapiro(data.sample(5000))
        else:
            sw_stat, sw_pvalue = stats.shapiro(data)
        
        # Test de Kolmogorov-Smirnov
        ks_stat, ks_pvalue = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        # Skewness et Kurtosis
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        results.append({
            'Asset': col,
            'JB_stat': jb_stat,
            'JB_pvalue': jb_pvalue,
            'JB_reject': jb_pvalue < alpha,
            'SW_stat': sw_stat,
            'SW_pvalue': sw_pvalue,
            'SW_reject': sw_pvalue < alpha,
            'KS_stat': ks_stat,
            'KS_pvalue': ks_pvalue,
            'Skewness': skew,
            'Kurtosis': kurt
        })
    
    return pd.DataFrame(results)

def plot_qq_plots(returns: pd.DataFrame, 
                  n_plots: int = 9,
                  figsize: Tuple[int, int] = (15, 10)):
    """
    Crée des QQ-plots pour visualiser les déviations de la normalité.
    
    Args:
        returns: DataFrame des rendements
        n_plots: Nombre de QQ-plots à afficher
        figsize: Taille de la figure
    """
    n_assets = min(n_plots, len(returns.columns))
    n_rows = int(np.ceil(np.sqrt(n_assets)))
    n_cols = int(np.ceil(n_assets / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_assets > 1 else [axes]
    
    for i, col in enumerate(returns.columns[:n_assets]):
        data = returns[col].dropna()
        stats.probplot(data, dist="norm", plot=axes[i])
        axes[i].set_title(f'{col}', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    # Cache les axes inutilisés
    for i in range(n_assets, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def test_stationarity(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Teste la stationnarité des séries avec le test ADF (Augmented Dickey-Fuller).
    
    Args:
        returns: DataFrame des rendements
        
    Returns:
        DataFrame avec les résultats
    """
    from statsmodels.tsa.stattools import adfuller
    
    results = []
    
    for col in returns.columns:
        data = returns[col].dropna()
        
        # Test ADF
        adf_result = adfuller(data, autolag='AIC')
        
        results.append({
            'Asset': col,
            'ADF_stat': adf_result[0],
            'ADF_pvalue': adf_result[1],
            'Stationary': adf_result[1] < 0.05,
            'Critical_1%': adf_result[4]['1%'],
            'Critical_5%': adf_result[4]['5%'],
            'Critical_10%': adf_result[4]['10%']
        })
    
    return pd.DataFrame(results)

def rolling_statistics(returns: pd.DataFrame,
                      window: int = 252) -> Dict[str, pd.DataFrame]:
    """
    Calcule les statistiques glissantes pour détecter la non-stationnarité.
    
    Args:
        returns: DataFrame des rendements
        window: Taille de la fenêtre (en jours)
        
    Returns:
        Dictionnaire avec les statistiques glissantes
    """
    rolling_mean = returns.rolling(window=window).mean() * 252
    rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    
    return {
        'mean': rolling_mean,
        'std': rolling_std,
        'sharpe': rolling_sharpe
    }

def plot_rolling_statistics(returns: pd.DataFrame,
                            asset: str,
                            window: int = 252,
                            figsize: Tuple[int, int] = (15, 10)):
    """
    Visualise les statistiques glissantes d'un actif.
    
    Args:
        returns: DataFrame des rendements
        asset: Nom de l'actif
        window: Taille de la fenêtre
        figsize: Taille de la figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    data = returns[asset]
    
    # Rendement moyen glissant
    rolling_mean = data.rolling(window=window).mean() * 252
    axes[0].plot(rolling_mean.index, rolling_mean.values, linewidth=2)
    axes[0].axhline(y=rolling_mean.mean(), color='r', linestyle='--', 
                   label='Mean overall')
    axes[0].set_ylabel('Rendement Annualisé')
    axes[0].set_title(f'{asset} - Statistiques Glissantes (fenêtre={window} jours)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Volatilité glissante
    rolling_std = data.rolling(window=window).std() * np.sqrt(252)
    axes[1].plot(rolling_std.index, rolling_std.values, linewidth=2, color='orange')
    axes[1].axhline(y=rolling_std.mean(), color='r', linestyle='--', 
                   label='Mean overall')
    axes[1].set_ylabel('Volatilité Annualisée')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Sharpe glissant
    rolling_sharpe = rolling_mean / rolling_std
    axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='green')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].axhline(y=rolling_sharpe.mean(), color='r', linestyle='--', 
                   label='Mean overall')
    axes[2].set_ylabel('Ratio de Sharpe')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def correlation_stability(returns: pd.DataFrame,
                          n_periods: int = 10) -> Dict:
    """
    Analyse la stabilité des corrélations dans le temps.
    
    Args:
        returns: DataFrame des rendements
        n_periods: Nombre de périodes à analyser
        
    Returns:
        Dictionnaire avec les matrices de corrélation par période
    """
    period_length = len(returns) // n_periods
    
    correlations = []
    periods = []
    
    for i in range(n_periods):
        start = i * period_length
        end = (i + 1) * period_length if i < n_periods - 1 else len(returns)
        
        period_returns = returns.iloc[start:end]
        corr = period_returns.corr()
        
        correlations.append(corr)
        periods.append(f"P{i+1}")
    
    # Calcule la stabilité moyenne (écart-type des corrélations)
    corr_stack = np.stack([c.values for c in correlations])
    corr_std = np.std(corr_stack, axis=0)
    
    return {
        'correlations': correlations,
        'periods': periods,
        'stability': pd.DataFrame(corr_std, index=returns.columns, columns=returns.columns),
        'mean_stability': corr_std[np.triu_indices_from(corr_std, k=1)].mean()
    }

def plot_correlation_stability(stability_results: Dict,
                               figsize: Tuple[int, int] = (12, 10)):
    """
    Visualise la stabilité des corrélations.
    
    Args:
        stability_results: Résultat de correlation_stability()
        figsize: Taille de la figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Première et dernière période
    sns.heatmap(stability_results['correlations'][0], 
                ax=axes[0, 0], cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.8})
    axes[0, 0].set_title(f"Corrélations - {stability_results['periods'][0]}")
    
    sns.heatmap(stability_results['correlations'][-1], 
                ax=axes[0, 1], cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.8})
    axes[0, 1].set_title(f"Corrélations - {stability_results['periods'][-1]}")
    
    # Écart-type des corrélations
    sns.heatmap(stability_results['stability'], 
                ax=axes[1, 0], cmap='Reds', 
                square=True, cbar_kws={"shrink": 0.8})
    axes[1, 0].set_title('Écart-type des Corrélations (instabilité)')
    
    # Distribution des écarts-types
    stab_values = stability_results['stability'].values
    stab_values = stab_values[np.triu_indices_from(stab_values, k=1)]
    
    axes[1, 1].hist(stab_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=stab_values.mean(), color='r', linestyle='--', 
                      label=f'Moyenne = {stab_values.mean():.3f}')
    axes[1, 1].set_xlabel('Écart-type des Corrélations')
    axes[1, 1].set_ylabel('Fréquence')
    axes[1, 1].set_title('Distribution de l\'Instabilité')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def estimation_error_analysis(returns: pd.DataFrame,
                              train_size: float = 0.8,
                              n_iterations: int = 100) -> Dict:
    """
    Analyse l'erreur d'estimation en comparant in-sample vs out-of-sample.
    
    Args:
        returns: DataFrame des rendements
        train_size: Proportion des données pour l'entraînement
        n_iterations: Nombre d'itérations
        
    Returns:
        Dictionnaire avec les erreurs d'estimation
    """
    n_train = int(len(returns) * train_size)
    
    mu_errors = []
    sigma_errors = []
    
    for _ in range(n_iterations):
        # Split aléatoire
        indices = np.random.permutation(len(returns))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_returns = returns.iloc[train_idx]
        test_returns = returns.iloc[test_idx]
        
        # Estimation sur train
        mu_train = train_returns.mean().values * 252
        Sigma_train = train_returns.cov().values * 252
        
        # Vérité sur test
        mu_test = test_returns.mean().values * 252
        Sigma_test = test_returns.cov().values * 252
        
        # Erreurs
        mu_error = np.mean(np.abs(mu_train - mu_test))
        sigma_error = np.mean(np.abs(Sigma_train - Sigma_test))
        
        mu_errors.append(mu_error)
        sigma_errors.append(sigma_error)
    
    return {
        'mu_error_mean': np.mean(mu_errors),
        'mu_error_std': np.std(mu_errors),
        'sigma_error_mean': np.mean(sigma_errors),
        'sigma_error_std': np.std(sigma_errors),
        'mu_errors': mu_errors,
        'sigma_errors': sigma_errors
    }

def concentration_analysis(weights: np.ndarray,
                          tickers: List[str],
                          ticker_sectors: Dict[str, str]) -> Dict:
    """
    Analyse la concentration d'un portefeuille.
    
    Args:
        weights: Vecteur de poids
        tickers: Liste des tickers
        ticker_sectors: Dictionnaire {ticker: secteur}
        
    Returns:
        Dictionnaire avec les métriques de concentration
    """
    from financial_metrics import sector_allocation
    
    # Concentration des actifs (indice de Herfindahl)
    herfindahl_assets = np.sum(weights ** 2)
    
    # Concentration sectorielle
    sector_weights = sector_allocation(weights, tickers, ticker_sectors)
    sector_weights_array = np.array(list(sector_weights.values()))
    herfindahl_sectors = np.sum(sector_weights_array ** 2)
    
    # Top N concentration
    sorted_weights = np.sort(weights)[::-1]
    top5_concentration = sorted_weights[:5].sum()
    top10_concentration = sorted_weights[:10].sum()
    
    # Entropie (diversification)
    weights_nonzero = weights[weights > 1e-6]
    entropy = -np.sum(weights_nonzero * np.log(weights_nonzero))
    
    return {
        'herfindahl_assets': herfindahl_assets,
        'herfindahl_sectors': herfindahl_sectors,
        'top5_concentration': top5_concentration,
        'top10_concentration': top10_concentration,
        'entropy': entropy,
        'effective_n_assets': 1 / herfindahl_assets,
        'effective_n_sectors': 1 / herfindahl_sectors,
        'sector_weights': sector_weights
    }

def generate_limits_report(returns: pd.DataFrame,
                          weights: np.ndarray,
                          tickers: List[str],
                          ticker_sectors: Dict[str, str]) -> Dict:
    """
    Génère un rapport complet sur les limites du modèle.
    
    Args:
        returns: DataFrame des rendements historiques
        weights: Vecteur de poids du portefeuille optimal
        tickers: Liste des tickers
        ticker_sectors: Mapping ticker->secteur
        
    Returns:
        Dictionnaire avec tous les résultats d'analyse
    """
    print("Analyse en cours...")
    
    # Tests de normalité
    print("  - Tests de normalité...")
    normality_results = test_normality(returns)
    n_reject_jb = normality_results['JB_reject'].sum()
    n_reject_sw = normality_results['SW_reject'].sum()
    
    # Tests de stationnarité
    print("  - Tests de stationnarité...")
    stationarity_results = test_stationarity(returns)
    n_stationary = stationarity_results['Stationary'].sum()
    
    # Stabilité des corrélations
    print("  - Analyse de stabilité des corrélations...")
    corr_stability = correlation_stability(returns, n_periods=5)
    
    # Erreurs d'estimation
    print("  - Analyse des erreurs d'estimation...")
    estimation_errors = estimation_error_analysis(returns, n_iterations=50)
    
    # Concentration
    print("  - Analyse de concentration...")
    concentration = concentration_analysis(weights, tickers, ticker_sectors)
    
    report = {
        'normality': {
            'results': normality_results,
            'n_assets': len(returns.columns),
            'n_reject_jb': n_reject_jb,
            'pct_reject_jb': n_reject_jb / len(returns.columns) * 100,
            'n_reject_sw': n_reject_sw,
            'pct_reject_sw': n_reject_sw / len(returns.columns) * 100
        },
        'stationarity': {
            'results': stationarity_results,
            'n_stationary': n_stationary,
            'pct_stationary': n_stationary / len(returns.columns) * 100
        },
        'correlation_stability': corr_stability,
        'estimation_errors': estimation_errors,
        'concentration': concentration
    }
    
    print("Analyse terminée!")
    return report

# Exemple d'utilisation
if __name__ == "__main__":
    # Génère des données synthétiques
    np.random.seed(42)
    n_assets = 20
    n_obs = 1000
    
    # Rendements avec queues épaisses (distribution t)
    returns = pd.DataFrame(
        stats.t.rvs(df=5, size=(n_obs, n_assets)) * 0.01,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    print("=== Test de Normalité ===")
    norm_results = test_normality(returns)
    print(f"Actifs rejetant la normalité (JB): {norm_results['JB_reject'].sum()}/{n_assets}")
    print(f"Actifs rejetant la normalité (SW): {norm_results['SW_reject'].sum()}/{n_assets}")
    
    print("\n=== Test de Stationnarité ===")
    stat_results = test_stationarity(returns)
    print(f"Actifs stationnaires: {stat_results['Stationary'].sum()}/{n_assets}")
    
    print("\n=== Stabilité des Corrélations ===")
    corr_stab = correlation_stability(returns, n_periods=5)
    print(f"Instabilité moyenne: {corr_stab['mean_stability']:.4f}")