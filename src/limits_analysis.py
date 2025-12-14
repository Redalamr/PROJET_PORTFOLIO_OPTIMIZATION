
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Dict

def test_normality(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Test de Jarque-Bera pour vérifier la normalité des rendements.
    Justifie la limite théorique du modèle Moyenne-Variance.
    """
    results = []
    for col in returns.columns:
        r = returns[col].dropna()
        # Jarque-Bera test
        jb_stat, jb_pvalue = stats.jarque_bera(r)
        # Shapiro-Wilk test (sur un sous-échantillon si trop grand)
        if len(r) > 5000:
            sw_stat, sw_pvalue = stats.shapiro(r.sample(5000))
        else:
            sw_stat, sw_pvalue = stats.shapiro(r)
            
        results.append({
            'Ticker': col,
            'JB_stat': jb_stat,
            'JB_pvalue': jb_pvalue,
            'JB_reject': jb_pvalue < 0.05,
            'SW_stat': sw_stat,
            'SW_pvalue': sw_pvalue,
            'SW_reject': sw_pvalue < 0.05,
            'Skewness': stats.skew(r),
            'Kurtosis': stats.kurtosis(r)
        })
    
    return pd.DataFrame(results).set_index('Ticker')

def plot_rolling_statistics(returns: pd.DataFrame, ticker: str, window: int = 252):
    """
    Affiche la moyenne et la volatilité glissantes.
    Justifie la non-stationnarité des données.
    """
    series = returns[ticker]
    rolling_mean = series.rolling(window=window).mean() * 252
    rolling_std = series.rolling(window=window).std() * np.sqrt(252)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rendement Moyen Annualisé', color=color)
    ax1.plot(series.index, rolling_mean, color=color, label='Moyenne Mobile')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Volatilité Annualisée', color=color) 
    ax2.plot(series.index, rolling_std, color=color, linestyle='--', label='Volatilité Mobile')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Stationnarité : Statistiques Glissantes ({window} jours) - {ticker}')
    fig.tight_layout()
    return fig

def test_stationarity(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Test ADF (Augmented Dickey-Fuller) pour la stationnarité.
    """
    from statsmodels.tsa.stattools import adfuller
    results = []
    for col in returns.columns:
        r = returns[col].dropna()
        try:
            adf_result = adfuller(r)
            results.append({
                'Ticker': col,
                'ADF_stat': adf_result[0],
                'ADF_pvalue': adf_result[1],
                'Stationary': adf_result[1] < 0.05
            })
        except:
            continue
            
    return pd.DataFrame(results).set_index('Ticker')

def estimation_error_analysis(returns: pd.DataFrame, n_iterations: int = 50, train_size: float = 0.5) -> dict:
    """
    Simule l'impact de l'échantillonnage sur les estimateurs (mu, Sigma).
    Justifie le besoin de robustesse.
    """
    mu_errors = []
    sigma_errors = []
    
    n_obs = len(returns)
    train_len = int(n_obs * train_size)
    
    # "Vrai" paramètres (sur tout l'historique)
    mu_true = returns.mean().values
    sigma_true = returns.cov().values
    
    for _ in range(n_iterations):
        # Échantillon aléatoire
        indices = np.random.choice(n_obs, train_len, replace=False)
        sample = returns.iloc[indices]
        
        mu_hat = sample.mean().values
        sigma_hat = sample.cov().values
        
        # Erreur relative (Norme L2)
        mu_errors.append(np.linalg.norm(mu_hat - mu_true) / np.linalg.norm(mu_true))
        sigma_errors.append(np.linalg.norm(sigma_hat - sigma_true) / np.linalg.norm(sigma_true))
        
    return {
        'mu_error_mean': np.mean(mu_errors),
        'mu_error_std': np.std(mu_errors),
        'sigma_error_mean': np.mean(sigma_errors),
        'sigma_error_std': np.std(sigma_errors),
        'mu_errors': mu_errors,
        'sigma_errors': sigma_errors
    }

def concentration_analysis(weights: np.ndarray, tickers: List[str], ticker_sectors: Dict[str, str]) -> dict:
    """
    Analyse la concentration sectorielle.
    Répond à la demande du PDF : "ventilation par types d'industrie".
    """
    df = pd.DataFrame({'Ticker': tickers, 'Weight': weights})
    df['Sector'] = df['Ticker'].map(ticker_sectors)
    
    # Poids par secteur
    sector_weights = df.groupby('Sector')['Weight'].sum()
    
    # Indice Herfindahl-Hirschman (HHI)
    hhi_sectors = (sector_weights**2).sum()
    hhi_assets = (weights**2).sum()
    
    # Top N concentration
    sorted_weights = np.sort(weights)[::-1]
    top5 = sorted_weights[:5].sum()
    top10 = sorted_weights[:10].sum()
    
    return {
        'sector_weights': sector_weights.to_dict(),
        'herfindahl_sectors': hhi_sectors,
        'herfindahl_assets': hhi_assets,
        'effective_n_sectors': 1/hhi_sectors if hhi_sectors > 0 else 0,
        'effective_n_assets': 1/hhi_assets if hhi_assets > 0 else 0,
        'top5_concentration': top5,
        'top10_concentration': top10
    }

# Fonctions supprimées car non demandées :
# - plot_qq_plots (optionnel)
# - correlation_stability (optionnel)
# - generate_limits_report (trop verbeux)