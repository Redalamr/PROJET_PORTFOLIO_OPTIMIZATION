"""
Optimisation par algorithme génétique NSGA-II (Niveau 2)
Problème tri-objectif avec contraintes de cardinalité et coûts de transaction
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_metrics import portfolio_return, portfolio_risk, transaction_costs


class PortfolioOptimizationProblem(Problem):
    """
    Problème d'optimisation de portefeuille multi-objectif.
    
    Objectifs:
        1. Minimiser -rendement (= maximiser rendement)
        2. Minimiser risque
        3. Minimiser coûts de transaction
    
    Contraintes:
        - Somme des poids = 1
        - Poids >= 0
        - Cardinalité = K (nombre fixe d'actifs)
    """
    
    def __init__(self, mu, Sigma, w_current=None, K=None, c_prop=0.005, delta_tol=1e-4):
        """
        Args:
            mu: Rendements moyens (N,)
            Sigma: Matrice de covariance (N, N)
            w_current: Portefeuille actuel (N,)
            K: Cardinalité cible (nombre d'actifs)
            c_prop: Coût proportionnel de transaction
            delta_tol: Seuil pour considérer un poids comme non-nul
        """
        self.mu = mu
        self.Sigma = Sigma
        self.n_assets = len(mu)
        self.w_current = w_current if w_current is not None else np.zeros(self.n_assets)
        self.K = K
        self.c_prop = c_prop
        self.delta_tol = delta_tol
        
        # Définit le problème pymoo
        super().__init__(
            n_var=self.n_assets,
            n_obj=3,  # 3 objectifs
            n_ieq_constr=1 if K is not None else 0,  # Contrainte de cardinalité
            xl=0.0,  # Borne inférieure
            xu=1.0   # Borne supérieure
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Évalue les objectifs et contraintes pour une population X.
        
        Args:
            X: Matrice (n_individuals, n_assets) des solutions
        """
        n_individuals = X.shape[0]
        
        # Normalise les poids pour que sum(w) = 1
        X_normalized = X / X.sum(axis=1, keepdims=True)
        
        # Calcule les objectifs
        f1 = np.zeros(n_individuals)  # -rendement
        f2 = np.zeros(n_individuals)  # risque
        f3 = np.zeros(n_individuals)  # coûts
        
        for i in range(n_individuals):
            w = X_normalized[i]
            f1[i] = -portfolio_return(w, self.mu)  # Minimiser -rendement
            f2[i] = portfolio_risk(w, self.Sigma)
            f3[i] = transaction_costs(w, self.w_current, self.c_prop)
        
        out["F"] = np.column_stack([f1, f2, f3])
        
        # Contrainte de cardinalité si spécifiée
        if self.K is not None:
            g = np.zeros(n_individuals)
            for i in range(n_individuals):
                w = X_normalized[i]
                cardinality = np.sum(w > self.delta_tol)
                # g <= 0 signifie contrainte respectée
                # On veut |cardinality - K| = 0
                g[i] = abs(cardinality - self.K)
            
            out["G"] = g.reshape(-1, 1)

class PortfolioOptimizationProblemBiObjective(Problem):
    """
    Version bi-objectif classique (sans coûts de transaction).
    """
    
    def __init__(self, mu, Sigma, K=None, delta_tol=1e-4):
        self.mu = mu
        self.Sigma = Sigma
        self.n_assets = len(mu)
        self.K = K
        self.delta_tol = delta_tol
        
        super().__init__(
            n_var=self.n_assets,
            n_obj=2,  # 2 objectifs seulement
            n_ieq_constr=1 if K is not None else 0,
            xl=0.0,
            xu=1.0
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        n_individuals = X.shape[0]
        X_normalized = X / X.sum(axis=1, keepdims=True)
        
        f1 = np.zeros(n_individuals)
        f2 = np.zeros(n_individuals)
        
        for i in range(n_individuals):
            w = X_normalized[i]
            f1[i] = -portfolio_return(w, self.mu)
            f2[i] = portfolio_risk(w, self.Sigma)
        
        out["F"] = np.column_stack([f1, f2])
        
        if self.K is not None:
            g = np.zeros(n_individuals)
            for i in range(n_individuals):
                w = X_normalized[i]
                cardinality = np.sum(w > self.delta_tol)
                g[i] = abs(cardinality - self.K)
            out["G"] = g.reshape(-1, 1)

def optimize_nsga2(mu, Sigma, w_current=None, K=None, c_prop=0.005, 
                   pop_size=100, n_gen=100, seed=42):
    """
    Optimise le portefeuille avec NSGA-II (tri-objectif).
    
    Args:
        mu: Rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        w_current: Portefeuille actuel (N,)
        K: Cardinalité cible
        c_prop: Coût proportionnel
        pop_size: Taille de la population
        n_gen: Nombre de générations
        seed: Graine aléatoire
        
    Returns:
        res: Objet résultat de pymoo contenant le front de Pareto
    """
    problem = PortfolioOptimizationProblem(
        mu=mu,
        Sigma=Sigma,
        w_current=w_current,
        K=K,
        c_prop=c_prop
    )
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", n_gen)
    
    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False
    )
    
    return res

def optimize_nsga2_biobjective(mu, Sigma, K=None, pop_size=100, n_gen=100, seed=42):
    """
    Optimise le portefeuille avec NSGA-II (bi-objectif, sans coûts).
    
    Args:
        mu: Rendements moyens (N,)
        Sigma: Matrice de covariance (N, N)
        K: Cardinalité cible
        pop_size: Taille de la population
        n_gen: Nombre de générations
        seed: Graine aléatoire
        
    Returns:
        res: Objet résultat de pymoo
    """
    problem = PortfolioOptimizationProblemBiObjective(
        mu=mu,
        Sigma=Sigma,
        K=K
    )
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", n_gen)
    
    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False
    )
    
    return res

def extract_pareto_front(res):
    """
    Extrait le front de Pareto d'un résultat NSGA-II.
    
    Args:
        res: Résultat de pymoo
        
    Returns:
        X: Solutions (poids des portefeuilles) normalisées
        F: Valeurs des objectifs
    """
    # Normalise les poids
    X = res.X / res.X.sum(axis=1, keepdims=True)
    F = res.F
    
    return X, F

def select_portfolio_from_front(X, F, min_return=None, max_risk=None, max_cost=None):
    """
    Sélectionne un portefeuille du front de Pareto selon des critères.
    
    Args:
        X: Matrice des solutions (n_solutions, n_assets)
        F: Matrice des objectifs (n_solutions, n_objectives)
        min_return: Rendement minimum requis
        max_risk: Risque maximum autorisé
        max_cost: Coût maximum autorisé
        
    Returns:
        idx: Indice du portefeuille sélectionné
        w: Vecteur de poids du portefeuille
    """
    # F[:,0] = -rendement, F[:,1] = risque, F[:,2] = coût (si tri-objectif)
    
    # Filtre selon les contraintes
    mask = np.ones(len(F), dtype=bool)
    
    if min_return is not None:
        mask &= (-F[:, 0] >= min_return)  # -F[:,0] = rendement
    
    if max_risk is not None:
        mask &= (F[:, 1] <= max_risk)
    
    if F.shape[1] >= 3 and max_cost is not None:
        mask &= (F[:, 2] <= max_cost)
    
    if not np.any(mask):
        raise ValueError("Aucun portefeuille ne satisfait les contraintes")
    
    # Parmi les candidats, sélectionne celui avec le meilleur compromis
    # Par exemple: minimise la somme normalisée des objectifs
    F_filtered = F[mask]
    X_filtered = X[mask]
    
    # Normalise chaque objectif entre 0 et 1
    F_norm = (F_filtered - F_filtered.min(axis=0)) / (F_filtered.max(axis=0) - F_filtered.min(axis=0) + 1e-10)
    
    # Sélectionne celui avec la plus petite somme normalisée
    scores = F_norm.sum(axis=1)
    best_idx_filtered = np.argmin(scores)
    
    # Trouve l'indice dans F original
    idx = np.where(mask)[0][best_idx_filtered]
    
    return idx, X[idx]