"""
Calculs des métriques financières pour l'optimisation de portefeuille
"""

import numpy as np
from typing import Tuple

def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """
    Calcule le rendement espéré du portefeuille.
    
    Args:
        w: Vecteur des poids (N,)
        mu: Vecteur des rendements moyens (N,)
        
    Returns:
        Rendement du portefeuille (scalaire)
    """
    return np.dot(w, mu)

def portfolio_risk(w: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Calcule le risque (variance) du portefeuille.
    
    Args:
        w: Vecteur des poids (N,)
        Sigma: Matrice de covariance (N, N)
        
    Returns:
        Risque du portefeuille (scalaire)
    """
    return np.dot(w, np.dot(Sigma, w))

def portfolio_volatility(w: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Calcule la volatilité (écart-type) du portefeuille.
    
    Args:
        w: Vecteur des poids (N,)
        Sigma: Matrice de covariance (N, N)
        
    Returns:
        Volatilité du portefeuille (scalaire)
    """
    return np.sqrt(portfolio_risk(w, Sigma))

def transaction_costs(w: np.ndarray, w_current: np.ndarray, c_prop: float = 0.005) -> float:
    """
    Calcule les coûts de transaction pour passer de w_current à w.
    
    Args:
        w: Nouveau vecteur de poids (N,)
        w_current: Vecteur de poids actuel (N,)
        c_prop: Coût proportionnel (par défaut 0.5%)
        
    Returns:
        Coût total de transaction (scalaire)
    """
    return c_prop * np.sum(np.abs(w - w_current))

def sharpe_ratio(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.02) -> float:
    """
    Calcule le ratio de Sharpe du portefeuille.
    
    Args:
        w: Vecteur des poids (N,)
        mu: Vecteur des rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        rf: Taux sans risque (par défaut 2%)
        
    Returns:
        Ratio de Sharpe (scalaire)
    """
    ret = portfolio_return(w, mu)
    vol = portfolio_volatility(w, Sigma)
    
    if vol == 0:
        return 0.0
    
    return (ret - rf) / vol

def portfolio_cardinality(w: np.ndarray, delta_tol: float = 1e-4) -> int:
    """
    Compte le nombre d'actifs effectivement présents dans le portefeuille.
    
    Args:
        w: Vecteur des poids (N,)
        delta_tol: Seuil en-dessous duquel un poids est considéré nul
        
    Returns:
        Nombre d'actifs actifs (entier)
    """
    return np.sum(w > delta_tol)

def evaluate_portfolio(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, 
                      w_current: np.ndarray = None, c_prop: float = 0.005,
                      delta_tol: float = 1e-4) -> Tuple[float, float, float, int]:
    """
    Évalue un portefeuille sur tous les critères.
    
    Args:
        w: Vecteur des poids (N,)
        mu: Vecteur des rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        w_current: Portefeuille actuel (si None, assume portefeuille vide)
        c_prop: Coût proportionnel
        delta_tol: Seuil de cardinalité
        
    Returns:
        return: Rendement
        risk: Risque (variance)
        cost: Coût de transaction
        card: Cardinalité
    """
    if w_current is None:
        w_current = np.zeros_like(w)
    
    ret = portfolio_return(w, mu)
    risk = portfolio_risk(w, Sigma)
    cost = transaction_costs(w, w_current, c_prop)
    card = portfolio_cardinality(w, delta_tol)
    
    return ret, risk, cost, card

def sector_allocation(w: np.ndarray, tickers: list, ticker_sectors: dict) -> dict:
    """
    Calcule la répartition du portefeuille par secteur.
    
    Args:
        w: Vecteur des poids (N,)
        tickers: Liste des noms de tickers
        ticker_sectors: Dictionnaire {ticker: secteur}
        
    Returns:
        Dictionnaire {secteur: poids_total}
    """
    sector_weights = {}
    
    for i, ticker in enumerate(tickers):
        if ticker in ticker_sectors:
            sector = ticker_sectors[ticker]
            if sector not in sector_weights:
                sector_weights[sector] = 0.0
            sector_weights[sector] += w[i]
    
    return sector_weights

def dominates(obj1: np.ndarray, obj2: np.ndarray, minimize: list = None) -> bool:
    """
    Vérifie si obj1 domine obj2 au sens de Pareto.
    
    Args:
        obj1: Vecteur d'objectifs 1
        obj2: Vecteur d'objectifs 2
        minimize: Liste de booléens indiquant si chaque objectif est à minimiser
        
    Returns:
        True si obj1 domine obj2
    """
    if minimize is None:
        minimize = [True] * len(obj1)
    
    # Convertit en problème de minimisation
    o1 = np.array([obj1[i] if minimize[i] else -obj1[i] for i in range(len(obj1))])
    o2 = np.array([obj2[i] if minimize[i] else -obj2[i] for i in range(len(obj2))])
    
    # obj1 domine obj2 si obj1 <= obj2 partout et obj1 < obj2 quelque part
    return np.all(o1 <= o2) and np.any(o1 < o2)

def is_pareto_efficient(costs: np.ndarray, return_mask: bool = True):
    """
    Trouve les solutions Pareto-efficaces dans un ensemble.
    
    Args:
        costs: Array (n_points, n_costs) à minimiser
        return_mask: Si True, retourne un masque booléen, sinon les indices
        
    Returns:
        Masque ou indices des solutions Pareto-efficaces
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Garde les points non dominés par c
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    
    if return_mask:
        return is_efficient
    else:
        return np.where(is_efficient)[0]