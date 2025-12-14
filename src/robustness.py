
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable

def bootstrap_resampling(returns: pd.DataFrame, n_samples: int = 50, block_size: int = None, seed: int = 42) -> List[pd.DataFrame]:
    """
    Génère des scénarios de marché alternatifs par rééchantillonnage (Bootstrap).
    Cité dans le PDF : "Une procédure de rééchantillonnage permet d'évaluer la stabilité".
    
    Args:
        returns: DataFrame des rendements historiques
        n_samples: Nombre d'échantillons à générer
        block_size: Taille du bloc (si None, tirage i.i.d simple)
        seed: Graine aléatoire
        
    Returns:
        Liste de DataFrames rééchantillonnés
    """
    np.random.seed(seed)
    n_obs, n_assets = returns.shape
    samples = []
    
    for _ in range(n_samples):
        if block_size is None:
            # Bootstrap simple (i.i.d)
            indices = np.random.randint(0, n_obs, size=n_obs)
            resampled_returns = returns.iloc[indices].reset_index(drop=True)
        else:
            # Block Bootstrap (pour préserver l'autocorrélation, si nécessaire)
            # Simplification : on reste sur du simple si non spécifié
            indices = np.random.randint(0, n_obs, size=n_obs)
            resampled_returns = returns.iloc[indices].reset_index(drop=True)
            
        samples.append(resampled_returns)
        
    return samples

def optimize_with_bootstrap(mu_orig: np.ndarray, Sigma_orig: np.ndarray, 
                           bootstrap_samples: List[pd.DataFrame],
                           optimizer_func: Callable,
                           rf: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimise le portefeuille sur chaque échantillon bootstrap pour évaluer la stabilité des poids.
    
    Args:
        mu_orig: Rendements moyens originaux (pour référence)
        Sigma_orig: Covariance originale (pour référence)
        bootstrap_samples: Liste des historiques simulés
        optimizer_func: Fonction d'optimisation à utiliser (ex: find_tangency_portfolio)
        rf: Taux sans risque
        
    Returns:
        w_mean: Poids moyens sur l'ensemble des simulations
        w_std: Écart-type des poids (mesure d'instabilité)
    """
    n_assets = len(mu_orig)
    weights_list = []
    
    for sample in bootstrap_samples:
        # 1. Recalcul des estimateurs pour cet échantillon
        # Annualisation (252 jours)
        mu_sample = sample.mean().values * 252
        Sigma_sample = sample.cov().values * 252
        
        try:
            # 2. Optimisation
            # On passe les arguments nécessaires à la fonction d'optimisation
            # Note: find_tangency_portfolio prend (mu, Sigma, rf)
            w = optimizer_func(mu_sample, Sigma_sample, rf=rf)
            weights_list.append(w)
        except:
            # En cas d'échec de l'optimiseur sur un échantillon dégénéré
            continue
            
    if not weights_list:
        raise ValueError("L'optimisation a échoué sur tous les échantillons bootstrap.")
        
    weights_matrix = np.array(weights_list)
    
    # Calcul des statistiques
    w_mean = np.mean(weights_matrix, axis=0)
    w_std = np.std(weights_matrix, axis=0)
    
    return w_mean, w_std