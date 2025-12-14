"""
Optimisation par algorithme génétique NSGA-II
Problème multi-objectif avec contrainte de cardinalité STRICTE (K=K)
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
import os, sys

# Ajustement du path pour importer financial_metrics si nécessaire
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_metrics import portfolio_return, portfolio_risk, transaction_costs

# =================================================================================
# 1. INITIALISATION INTELLIGENTE (SAMPLING)
# =================================================================================

class PortfolioSampling(Sampling):
    """
    Initialisation personnalisée qui génère des portefeuilles
    contenant EXACTEMENT K actifs (les autres sont à 0).
    """
    def __init__(self, K=None):
        super().__init__()
        self.K = K

    def _do(self, problem, n_samples, **kwargs):
        n_assets = problem.n_var
        # 1. Génère une matrice aléatoire (n_samples, n_assets)
        X = np.random.random((n_samples, n_assets))

        # 2. Si une contrainte K existe, on force les zéros
        if self.K is not None and self.K < n_assets:
            for i in range(n_samples):
                # On choisit aléatoirement K indices à conserver
                active_indices = np.random.choice(n_assets, self.K, replace=False)
                
                # Masque : True pour les actifs gardés, False pour les autres
                mask = np.zeros(n_assets, dtype=bool)
                mask[active_indices] = True
                
                # Mise à zéro stricte des actifs non sélectionnés
                X[i, ~mask] = 0.0
        
        # 3. Normalisation (Somme des poids = 1)
        row_sums = X.sum(axis=1, keepdims=True)
        # Sécurité pour éviter division par zéro (si jamais le générateur fait des zéros partout)
        row_sums[row_sums == 0] = 1.0 
        
        return X / row_sums


# =================================================================================
# 2. DÉFINITION DES PROBLÈMES PYMOO
# =================================================================================

class PortfolioOptimizationProblem(Problem):
    """
    Problème Tri-Objectif : Rendement, Risque, Coûts de transaction.
    """
    def __init__(self, mu, Sigma, w_current=None, K=None, c_prop=0.005, delta_tol=1e-4):
        self.mu = mu
        self.Sigma = Sigma
        self.n_assets = len(mu)
        self.w_current = w_current if w_current is not None else np.zeros(self.n_assets)
        self.K = K
        self.c_prop = c_prop
        self.delta_tol = delta_tol
        
        super().__init__(
            n_var=self.n_assets,
            n_obj=3,  # 3 objectifs
            n_ieq_constr=1 if K is not None else 0, # 1 contrainte si K défini
            xl=0.0,
            xu=1.0
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        n_individuals = X.shape[0]
        
        # Normalisation systématique avant évaluation
        X_normalized = X / X.sum(axis=1, keepdims=True)
        
        f1 = np.zeros(n_individuals) # -Rendement
        f2 = np.zeros(n_individuals) # Risque
        f3 = np.zeros(n_individuals) # Coûts
        
        for i in range(n_individuals):
            w = X_normalized[i]
            f1[i] = -portfolio_return(w, self.mu)
            f2[i] = portfolio_risk(w, self.Sigma)
            f3[i] = transaction_costs(w, self.w_current, self.c_prop)
        
        out["F"] = np.column_stack([f1, f2, f3])
        
        # Contrainte de Cardinalité STRICTE
        if self.K is not None:
            g = np.zeros(n_individuals)
            for i in range(n_individuals):
                w = X_normalized[i]
                # Compte le nombre d'actifs > delta_tol (ex: 0.0001)
                cardinality = np.sum(w > self.delta_tol)
                
                # Contrainte : |cardinality - K| <= 0
                # Si card != K, g sera positif (donc contrainte violée)
                g[i] = abs(cardinality - self.K)
            
            out["G"] = g.reshape(-1, 1)


class PortfolioOptimizationProblemBiObjective(Problem):
    """
    Problème Bi-Objectif : Rendement, Risque (Sans coûts).
    """
    def __init__(self, mu, Sigma, K=None, delta_tol=1e-4):
        self.mu = mu
        self.Sigma = Sigma
        self.n_assets = len(mu)
        self.K = K
        self.delta_tol = delta_tol
        
        super().__init__(
            n_var=self.n_assets,
            n_obj=2,  # 2 objectifs
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
        
        # Contrainte de Cardinalité STRICTE
        if self.K is not None:
            g = np.zeros(n_individuals)
            for i in range(n_individuals):
                w = X_normalized[i]
                cardinality = np.sum(w > self.delta_tol)
                # Ecart strict par rapport à K
                g[i] = abs(cardinality - self.K)
            out["G"] = g.reshape(-1, 1)


# =================================================================================
# 3. FONCTIONS D'OPTIMISATION (APPELS UTILISATEUR)
# =================================================================================

def optimize_nsga2(mu, Sigma, w_current=None, K=None, c_prop=0.005, 
                   pop_size=100, n_gen=100, seed=42):
    """
    Lance l'optimisation Tri-Objectif.
    """
    problem = PortfolioOptimizationProblem(
        mu=mu, Sigma=Sigma, w_current=w_current, K=K, c_prop=c_prop
    )
    
    # Utilisation du Sampling personnalisé
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PortfolioSampling(K=K),  # <--- FORCE L'INITIALISATION CORRECTE
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
    Lance l'optimisation Bi-Objectif.
    """
    problem = PortfolioOptimizationProblemBiObjective(
        mu=mu, Sigma=Sigma, K=K
    )
    
    # Utilisation du Sampling personnalisé
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PortfolioSampling(K=K),  # <--- FORCE L'INITIALISATION CORRECTE
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


# =================================================================================
# 4. FONCTIONS UTILITAIRES
# =================================================================================

def extract_pareto_front(res):
    """
    Extrait les solutions du résultat.
    Gère les erreurs si aucune solution n'est trouvée.
    """
    if res.X is None or res.F is None:
        raise ValueError(
            "L'optimisation NSGA-II a échoué. Aucune solution trouvée.\n"
            "Suggestions:\n"
            "  - Augmentez pop_size (ex: 200) et n_gen (ex: 200)\n"
            "  - Vérifiez vos données (mu, Sigma)"
        )
    
    # Normalisation finale pour être sûr
    X = res.X / res.X.sum(axis=1, keepdims=True)
    F = res.F
    
    return X, F


def select_portfolio_from_front(X, F, min_return=None, max_risk=None, max_cost=None):
    """
    Sélectionne un portefeuille spécifique sur le front de Pareto.
    """
    mask = np.ones(len(F), dtype=bool)
    
    if min_return is not None:
        mask &= (-F[:, 0] >= min_return)
    
    if max_risk is not None:
        mask &= (F[:, 1] <= max_risk)
    
    if F.shape[1] >= 3 and max_cost is not None:
        mask &= (F[:, 2] <= max_cost)
    
    if not np.any(mask):
        raise ValueError("Aucun portefeuille ne satisfait les contraintes demandées")
    
    F_filtered = F[mask]
    
    # Normalisation Min-Max pour score de compromis
    F_min = F_filtered.min(axis=0)
    F_max = F_filtered.max(axis=0)
    denom = F_max - F_min
    denom[denom == 0] = 1e-10 # Évite division par zéro
    
    F_norm = (F_filtered - F_min) / denom
    
    # On cherche le "meilleur compromis" (distance minimale à l'origine normalisée)
    scores = np.sum(F_norm**2, axis=1) # Distance euclidienne
    best_idx_filtered = np.argmin(scores)
    
    # Retrouver l'indice original
    idx = np.where(mask)[0][best_idx_filtered]
    
    return idx, X[idx]