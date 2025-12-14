"""
Méthodes d'analyse de robustesse pour l'optimisation de portefeuille
À placer dans : src/robustness.py
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Callable
from scipy.linalg import inv
import cvxpy as cp

def black_litterman(Sigma: np.ndarray, 
                   market_caps: np.ndarray = None,
                   P: np.ndarray = None, 
                   Q: np.ndarray = None,
                   tau: float = 0.025,
                   omega: np.ndarray = None,
                   risk_aversion: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modèle de Black-Litterman pour estimer les rendements attendus.
    
    Args:
        Sigma: Matrice de covariance (N, N)
        market_caps: Capitalisations de marché (poids d'équilibre) (N,)
        P: Matrice des vues (K, N) - K vues sur N actifs
        Q: Vecteur des rendements prévus pour les vues (K,)
        tau: Facteur d'incertitude (typiquement 0.025)
        omega: Matrice de covariance de l'incertitude des vues (K, K)
        risk_aversion: Coefficient d'aversion au risque
        
    Returns:
        mu_bl: Rendements espérés Black-Litterman (N,)
        Sigma_bl: Matrice de covariance ajustée (N, N)
    """
    N = Sigma.shape[0]
    
    # Si pas de poids de marché, assume équipondéré
    if market_caps is None:
        market_caps = np.ones(N) / N
    
    # Rendements implicites d'équilibre
    pi = risk_aversion * Sigma @ market_caps
    
    # Si pas de vues, retourne les rendements d'équilibre
    if P is None or Q is None:
        return pi, Sigma
    
    K = P.shape[0]
    
    # Si pas d'incertitude sur les vues, assume proportionnelle à P Sigma P'
    if omega is None:
        omega = np.diag(np.diag(P @ Sigma @ P.T)) * tau
    
    # Formules de Black-Litterman
    # Calcul de la matrice de précision
    tau_Sigma = tau * Sigma
    
    # Inverse de la matrice de variance des vues
    omega_inv = inv(omega)
    
    # Rendements attendus ajustés
    M = inv(inv(tau_Sigma) + P.T @ omega_inv @ P)
    mu_bl = M @ (inv(tau_Sigma) @ pi + P.T @ omega_inv @ Q)
    
    # Matrice de covariance ajustée
    Sigma_bl = Sigma + M
    
    return mu_bl, Sigma_bl

def create_view_matrix(n_assets: int, 
                      view_type: str, 
                      assets_indices: List[int],
                      reference_index: int = None) -> Tuple[np.ndarray, float]:
    """
    Crée une matrice de vue pour Black-Litterman.
    
    Args:
        n_assets: Nombre total d'actifs
        view_type: 'absolute' ou 'relative'
        assets_indices: Indices des actifs concernés
        reference_index: Indice de l'actif de référence (pour vue relative)
        
    Returns:
        P: Ligne de la matrice de vue (1, N)
        Q: Rendement prévu (scalaire)
    """
    P = np.zeros((1, n_assets))
    
    if view_type == 'absolute':
        # Vue absolue : "L'actif i aura un rendement de Q%"
        P[0, assets_indices[0]] = 1.0
        
    elif view_type == 'relative':
        # Vue relative : "L'actif i surperformera l'actif j de Q%"
        P[0, assets_indices[0]] = 1.0
        P[0, reference_index] = -1.0
    
    return P

def bootstrap_resampling(returns: pd.DataFrame, 
                        n_samples: int = 1000,
                        block_size: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Génère des échantillons bootstrap des paramètres (mu, Sigma).
    Utilise le block bootstrap pour préserver l'autocorrélation.
    
    Args:
        returns: DataFrame des rendements historiques
        n_samples: Nombre d'échantillons bootstrap
        block_size: Taille des blocs (pour préserver l'autocorrélation)
        
    Returns:
        Liste de tuples (mu_bootstrap, Sigma_bootstrap)
    """
    n_obs = len(returns)
    n_blocks = n_obs // block_size
    
    samples = []
    
    for _ in range(n_samples):
        # Échantillonnage des blocs
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        
        # Reconstruction de la série
        bootstrap_returns = []
        for block_idx in block_indices:
            start = block_idx * block_size
            end = min(start + block_size, n_obs)
            bootstrap_returns.append(returns.iloc[start:end])
        
        bootstrap_returns = pd.concat(bootstrap_returns, ignore_index=True)
        
        # Calcul des statistiques
        mu_boot = bootstrap_returns.mean().values * 252
        Sigma_boot = bootstrap_returns.cov().values * 252
        
        samples.append((mu_boot, Sigma_boot))
    
    return samples

def optimize_with_bootstrap(mu: np.ndarray,
                            Sigma: np.ndarray,
                            bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]],
                            optimizer_func: Callable,
                            **optimizer_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimise un portefeuille sur plusieurs échantillons bootstrap.
    
    Args:
        mu: Rendements moyens de référence
        Sigma: Covariance de référence
        bootstrap_samples: Liste de (mu_boot, Sigma_boot)
        optimizer_func: Fonction d'optimisation (ex: markowitz_min_risk)
        optimizer_kwargs: Arguments pour l'optimiseur
        
    Returns:
        w_mean: Poids moyens sur tous les échantillons
        w_std: Écart-type des poids
    """
    n_assets = len(mu)
    n_samples = len(bootstrap_samples)
    
    weights_matrix = np.zeros((n_samples, n_assets))
    
    for i, (mu_boot, Sigma_boot) in enumerate(bootstrap_samples):
        try:
            w = optimizer_func(mu_boot, Sigma_boot, **optimizer_kwargs)
            weights_matrix[i] = w
        except:
            # Si l'optimisation échoue, utilise les paramètres de référence
            w = optimizer_func(mu, Sigma, **optimizer_kwargs)
            weights_matrix[i] = w
    
    w_mean = weights_matrix.mean(axis=0)
    w_std = weights_matrix.std(axis=0)
    
    # Renormalise les poids moyens
    w_mean = w_mean / w_mean.sum()
    
    return w_mean, w_std

def robust_markowitz(mu: np.ndarray,
                    Sigma: np.ndarray,
                    uncertainty_mu: np.ndarray = None,
                    uncertainty_Sigma: np.ndarray = None,
                    gamma: float = 0.5) -> np.ndarray:
    """
    Optimisation robuste de Markowitz (worst-case approach).
    Minimise le risque dans le pire des cas.
    
    Args:
        mu: Rendements moyens
        Sigma: Matrice de covariance
        uncertainty_mu: Incertitude sur mu (écart-type) (N,)
        uncertainty_Sigma: Incertitude sur Sigma (N, N)
        gamma: Paramètre de robustesse (0=non robuste, 1=très robuste)
        
    Returns:
        Vecteur de poids optimal
    """
    n = len(mu)
    
    # Si pas d'incertitude spécifiée, estime à partir de la diagonale de Sigma
    if uncertainty_mu is None:
        uncertainty_mu = np.sqrt(np.diag(Sigma)) / np.sqrt(252)  # Erreur standard annualisée
    
    if uncertainty_Sigma is None:
        uncertainty_Sigma = Sigma * 0.1  # 10% d'incertitude
    
    w = cp.Variable(n)
    
    # Rendement worst-case
    ret_worst_case = mu @ w - gamma * cp.norm(uncertainty_mu * w, 2)
    
    # Risque worst-case
    risk_worst_case = cp.quad_form(w, Sigma + gamma * uncertainty_Sigma)
    
    # Objectif : maximiser Sharpe worst-case (approximation)
    # On minimise -ret_worst_case / sqrt(risk_worst_case)
    # Équivalent à minimiser risk_worst_case sous contrainte de rendement
    
    objective = cp.Minimize(risk_worst_case)
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        ret_worst_case >= mu.mean() * 0.5  # Rendement minimum
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)
    
    return w.value

def sensitivity_analysis(mu: np.ndarray,
                        Sigma: np.ndarray,
                        optimizer_func: Callable,
                        param_name: str = 'mu',
                        perturbation_range: np.ndarray = None,
                        **optimizer_kwargs) -> dict:
    """
    Analyse de sensibilité : varie un paramètre et observe les changements.
    
    Args:
        mu: Rendements moyens de référence
        Sigma: Covariance de référence
        optimizer_func: Fonction d'optimisation
        param_name: 'mu' ou 'Sigma'
        perturbation_range: Range des perturbations (ex: [-0.05, 0, 0.05])
        optimizer_kwargs: Arguments pour l'optimiseur
        
    Returns:
        Dictionnaire avec les résultats
    """
    if perturbation_range is None:
        perturbation_range = np.linspace(-0.1, 0.1, 11)
    
    results = {
        'perturbations': [],
        'weights': [],
        'returns': [],
        'risks': [],
        'weight_changes': []
    }
    
    # Portefeuille de référence (sans perturbation)
    w_ref = optimizer_func(mu, Sigma, **optimizer_kwargs)
    
    for delta in perturbation_range:
        if param_name == 'mu':
            # Perturbe mu uniformément
            mu_perturbed = mu * (1 + delta)
            Sigma_perturbed = Sigma
        elif param_name == 'Sigma':
            # Perturbe Sigma uniformément
            mu_perturbed = mu
            Sigma_perturbed = Sigma * (1 + delta)
        else:
            raise ValueError("param_name doit être 'mu' ou 'Sigma'")
        
        try:
            w = optimizer_func(mu_perturbed, Sigma_perturbed, **optimizer_kwargs)
            
            results['perturbations'].append(delta)
            results['weights'].append(w)
            results['returns'].append(w @ mu_perturbed)
            results['risks'].append(w @ Sigma_perturbed @ w)
            results['weight_changes'].append(np.sum(np.abs(w - w_ref)))
        except:
            continue
    
    return results

def resampled_efficient_frontier(returns: pd.DataFrame,
                                 n_bootstrap: int = 100,
                                 n_points: int = 20) -> dict:
    """
    Génère une frontière efficace réechantillonnée (Michaud).
    
    Args:
        returns: DataFrame des rendements historiques
        n_bootstrap: Nombre d'échantillons bootstrap
        n_points: Nombre de points sur la frontière
        
    Returns:
        Dictionnaire avec les frontières
    """
    from optimizers.classic import generate_efficient_frontier_scalarization
    
    # Paramètres de référence
    mu_ref = returns.mean().values * 252
    Sigma_ref = returns.cov().values * 252
    
    # Frontière de référence
    portfolios_ref, rets_ref, risks_ref = generate_efficient_frontier_scalarization(
        mu_ref, Sigma_ref, n_points=n_points
    )
    
    # Bootstrap
    bootstrap_samples = bootstrap_resampling(returns, n_samples=n_bootstrap)
    
    # Frontières bootstrap
    all_rets = []
    all_risks = []
    
    for mu_boot, Sigma_boot in bootstrap_samples:
        try:
            _, rets, risks = generate_efficient_frontier_scalarization(
                mu_boot, Sigma_boot, n_points=n_points
            )
            all_rets.append(rets)
            all_risks.append(risks)
        except:
            continue
    
    all_rets = np.array(all_rets)
    all_risks = np.array(all_risks)
    
    # Statistiques
    rets_mean = all_rets.mean(axis=0)
    rets_std = all_rets.std(axis=0)
    risks_mean = all_risks.mean(axis=0)
    risks_std = all_risks.std(axis=0)
    
    return {
        'reference': {
            'returns': rets_ref,
            'risks': risks_ref,
            'portfolios': portfolios_ref
        },
        'bootstrap': {
            'returns_mean': rets_mean,
            'returns_std': rets_std,
            'risks_mean': risks_mean,
            'risks_std': risks_std,
            'all_returns': all_rets,
            'all_risks': all_risks
        }
    }

def calculate_var_cvar(returns: pd.DataFrame,
                      weights: np.ndarray,
                      confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcule la Value-at-Risk et la Conditional VaR (Expected Shortfall).
    
    Args:
        returns: DataFrame des rendements historiques
        weights: Vecteur de poids du portefeuille
        confidence: Niveau de confiance (ex: 0.95 pour 95%)
        
    Returns:
        VaR, CVaR (tous deux positifs, en %)
    """
    # Rendements du portefeuille
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # VaR = quantile
    var = -np.percentile(portfolio_returns, (1 - confidence) * 100) * 100
    
    # CVaR = moyenne des pertes au-delà de la VaR
    losses = -portfolio_returns[portfolio_returns < -var/100]
    cvar = losses.mean() * 100 if len(losses) > 0 else var
    
    return var, cvar

# Exemple d'utilisation
if __name__ == "__main__":
    # Génère des données synthétiques
    np.random.seed(42)
    n_assets = 10
    n_obs = 1000
    
    returns = pd.DataFrame(
        np.random.randn(n_obs, n_assets) * 0.01,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    mu = returns.mean().values * 252
    Sigma = returns.cov().values * 252
    
    print("=== Black-Litterman ===")
    # Vue : Asset_0 aura un rendement de 15%
    P = create_view_matrix(n_assets, 'absolute', [0])[0]
    Q = np.array([0.15])
    
    mu_bl, Sigma_bl = black_litterman(Sigma, P=P, Q=Q)
    print(f"Rendement original Asset_0: {mu[0]:.4f}")
    print(f"Rendement BL Asset_0: {mu_bl[0]:.4f}")
    
    print("\n=== Bootstrap Resampling ===")
    bootstrap_samples = bootstrap_resampling(returns, n_samples=10)
    print(f"Généré {len(bootstrap_samples)} échantillons bootstrap")
    
    print("\n=== Analyse de Sensibilité ===")
    from optimizers.classic import find_tangency_portfolio
    
    sens_results = sensitivity_analysis(
        mu, Sigma, find_tangency_portfolio,
        param_name='mu',
        perturbation_range=np.linspace(-0.1, 0.1, 5)
    )
    
    print("Changements de poids avec perturbations de mu:")
    for delta, change in zip(sens_results['perturbations'], sens_results['weight_changes']):
        print(f"  δ={delta:+.2f}: Changement total = {change:.4f}")