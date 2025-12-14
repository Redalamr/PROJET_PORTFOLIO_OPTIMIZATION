"""
Optimisation classique de Markowitz (Niveau 1)
Problème bi-objectif: minimiser risque et maximiser rendement
"""

import numpy as np
from scipy.optimize import minimize
import cvxpy as cp
from typing import Tuple, List
import sys
sys.path.append('..')
from financial_metrics import portfolio_return, portfolio_risk

def markowitz_min_risk(mu: np.ndarray, Sigma: np.ndarray, target_return: float = None) -> np.ndarray:
    """
    Résout le problème de Markowitz: minimiser le risque pour un rendement cible.
    
    Args:
        mu: Vecteur des rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        target_return: Rendement cible (si None, pas de contrainte)
        
    Returns:
        Vecteur de poids optimal (N,)
    """
    n = len(mu)
    w = cp.Variable(n)
    
    # Objectif: minimiser le risque
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    
    # Contraintes de base
    constraints = [
        cp.sum(w) == 1,  # Investissement complet
        w >= 0            # Pas de vente à découvert
    ]
    
    # Contrainte de rendement cible
    if target_return is not None:
        constraints.append(mu @ w >= target_return)
    
    # Résolution
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)
    
    if w.value is None:
        raise ValueError("Optimisation échouée")
    
    return w.value

def markowitz_max_return(mu: np.ndarray, Sigma: np.ndarray, max_risk: float) -> np.ndarray:
    """
    Résout le problème de Markowitz: maximiser le rendement pour un risque maximum.
    
    Args:
        mu: Vecteur des rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        max_risk: Risque maximum autorisé
        
    Returns:
        Vecteur de poids optimal (N,)
    """
    n = len(mu)
    w = cp.Variable(n)
    
    # Objectif: maximiser le rendement
    objective = cp.Maximize(mu @ w)
    
    # Contraintes
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.quad_form(w, Sigma) <= max_risk
    ]
    
    # Résolution
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)
    
    if w.value is None:
        raise ValueError("Optimisation échouée")
    
    return w.value

def scalarization_weighted_sum(mu: np.ndarray, Sigma: np.ndarray, alpha: float) -> np.ndarray:
    """
    Méthode de scalarisation par somme pondérée.
    Minimise: alpha * risque - (1-alpha) * rendement
    
    Args:
        mu: Vecteur des rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        alpha: Poids du risque dans [0, 1] (0=max rendement, 1=min risque)
        
    Returns:
        Vecteur de poids optimal (N,)
    """
    n = len(mu)
    w = cp.Variable(n)
    
    # Objectif scalarisé
    risk = cp.quad_form(w, Sigma)
    ret = mu @ w
    objective = cp.Minimize(alpha * risk - (1 - alpha) * ret)
    
    # Contraintes
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Résolution
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)
    
    if w.value is None:
        raise ValueError("Optimisation échouée")
    
    return w.value

def epsilon_constraint(mu: np.ndarray, Sigma: np.ndarray, min_return: float) -> np.ndarray:
    """
    Méthode epsilon-contrainte: minimise le risque sous contrainte de rendement minimum.
    
    Args:
        mu: Vecteur des rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        min_return: Rendement minimum requis
        
    Returns:
        Vecteur de poids optimal (N,)
    """
    return markowitz_min_risk(mu, Sigma, target_return=min_return)

def generate_efficient_frontier_scalarization(mu: np.ndarray, Sigma: np.ndarray, 
                                              n_points: int = 50) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Génère la frontière efficace par scalarisation.
    
    Args:
        mu: Vecteur des rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        n_points: Nombre de points sur la frontière
        
    Returns:
        portfolios: Liste des vecteurs de poids
        returns: Vecteurs des rendements
        risks: Vecteur des risques
    """
    portfolios = []
    returns = []
    risks = []
    
    # Génère des valeurs alpha uniformément réparties
    alphas = np.linspace(0, 1, n_points)
    
    for alpha in alphas:
        try:
            w = scalarization_weighted_sum(mu, Sigma, alpha)
            portfolios.append(w)
            returns.append(portfolio_return(w, mu))
            risks.append(portfolio_risk(w, Sigma))
        except:
            continue
    
    return portfolios, np.array(returns), np.array(risks)

def generate_efficient_frontier_epsilon(mu: np.ndarray, Sigma: np.ndarray, 
                                       n_points: int = 50) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Génère la frontière efficace par méthode epsilon-contrainte.
    
    Args:
        mu: Vecteur des rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        n_points: Nombre de points sur la frontière
        
    Returns:
        portfolios: Liste des vecteurs de poids
        returns: Vecteurs des rendements
        risks: Vecteur des risques
    """
    portfolios = []
    returns = []
    risks = []
    
    # Calcule les rendements min et max
    min_return = np.min(mu)
    max_return = np.max(mu)
    
    # Génère des rendements cibles
    target_returns = np.linspace(min_return, max_return, n_points)
    
    for target_ret in target_returns:
        try:
            w = epsilon_constraint(mu, Sigma, target_ret)
            portfolios.append(w)
            returns.append(portfolio_return(w, mu))
            risks.append(portfolio_risk(w, Sigma))
        except:
            continue
    
    return portfolios, np.array(returns), np.array(risks)

def find_tangency_portfolio(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.02) -> np.ndarray:
    """
    Trouve le portefeuille tangent (Sharpe ratio maximum).
    
    Args:
        mu: Vecteur des rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        rf: Taux sans risque
        
    Returns:
        Vecteur de poids optimal (N,)
    """
    n = len(mu)
    
    # Résolution analytique: w = Sigma^(-1) * (mu - rf)
    try:
        excess_returns = mu - rf
        Sigma_inv = np.linalg.inv(Sigma)
        w = Sigma_inv @ excess_returns
        w = w / np.sum(w)  # Normalisation
        
        # Projette sur le simplexe (poids positifs)
        w = np.maximum(w, 0)
        if np.sum(w) > 0:
            w = w / np.sum(w)
        else:
            # Si tous négatifs, résout par optimisation
            w_var = cp.Variable(n)
            sharpe = (mu @ w_var - rf) / cp.sqrt(cp.quad_form(w_var, Sigma))
            objective = cp.Maximize(sharpe)
            constraints = [cp.sum(w_var) == 1, w_var >= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP)
            w = w_var.value
            
    except:
        # Méthode d'optimisation si inverse échoue
        w_var = cp.Variable(n)
        sharpe = (mu @ w_var - rf) / cp.sqrt(cp.quad_form(w_var, Sigma))
        objective = cp.Maximize(sharpe)
        constraints = [cp.sum(w_var) == 1, w_var >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)
        w = w_var.value
    
    return w

def find_minimum_variance_portfolio(Sigma: np.ndarray) -> np.ndarray:
    """
    Trouve le portefeuille de variance minimale globale.
    
    Args:
        Sigma: Matrice de covariance (N, N)
        
    Returns:
        Vecteur de poids optimal (N,)
    """
    n = Sigma.shape[0]
    w = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='ECOS')
    
    return w.value