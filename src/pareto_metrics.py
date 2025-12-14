"""
Métriques de qualité pour l'évaluation des fronts de Pareto
À placer dans : src/pareto_metrics.py
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, List

def normalize_objectives(F: np.ndarray) -> np.ndarray:
    """
    Normalise les objectifs entre 0 et 1.
    
    Args:
        F: Matrice (n_solutions, n_objectives)
        
    Returns:
        Matrice normalisée
    """
    F_min = F.min(axis=0)
    F_max = F.max(axis=0)
    F_range = F_max - F_min
    F_range[F_range == 0] = 1  # Évite division par zéro
    
    return (F - F_min) / F_range

def hypervolume_2d(F: np.ndarray, ref_point: np.ndarray = None) -> float:
    """
    Calcule l'hypervolume pour un front 2D (méthode exacte).
    
    Args:
        F: Matrice (n_solutions, 2) des objectifs à MINIMISER
        ref_point: Point de référence (si None, utilise max + 0.1)
        
    Returns:
        Hypervolume (plus grand = meilleur)
    """
    # Normalise
    F_norm = normalize_objectives(F)
    
    # Point de référence
    if ref_point is None:
        ref_point = F_norm.max(axis=0) + 0.1
    
    # Trie par premier objectif
    indices = np.argsort(F_norm[:, 0])
    F_sorted = F_norm[indices]
    
    # Calcul de l'hypervolume par méthode des rectangles
    hv = 0.0
    prev_x = 0.0
    
    for i in range(len(F_sorted)):
        x = F_sorted[i, 0]
        y = F_sorted[i, 1]
        
        # Hauteur du rectangle
        height = ref_point[1] - y
        # Largeur du rectangle
        width = x - prev_x
        
        hv += width * height
        prev_x = x
    
    # Dernier rectangle jusqu'au point de référence
    if len(F_sorted) > 0:
        last_x = F_sorted[-1, 0]
        last_y = F_sorted[-1, 1]
        hv += (ref_point[0] - last_x) * (ref_point[1] - last_y)
    
    return hv

def hypervolume_3d(F: np.ndarray, ref_point: np.ndarray = None) -> float:
    """
    Calcule l'hypervolume pour un front 3D (approximation par Monte Carlo).
    
    Args:
        F: Matrice (n_solutions, 3) des objectifs à MINIMISER
        ref_point: Point de référence
        
    Returns:
        Hypervolume approximé
    """
    F_norm = normalize_objectives(F)
    
    if ref_point is None:
        ref_point = F_norm.max(axis=0) + 0.1
    
    # Monte Carlo sampling
    n_samples = 100000
    samples = np.random.uniform(
        low=F_norm.min(axis=0),
        high=ref_point,
        size=(n_samples, 3)
    )
    
    # Compte les points dominés par au moins une solution du front
    dominated = np.zeros(n_samples, dtype=bool)
    
    for sol in F_norm:
        # Un point est dominé si toutes ses coordonnées sont >= à celles de sol
        dominated |= np.all(samples >= sol, axis=1)
    
    # Volume de la boîte
    box_volume = np.prod(ref_point - F_norm.min(axis=0))
    
    # Proportion de points dominés
    hv = box_volume * (1 - dominated.sum() / n_samples)
    
    return hv

def spacing_metric(F: np.ndarray) -> float:
    """
    Calcule la métrique de spacing (uniformité de la distribution).
    Plus petit = meilleur (solutions plus uniformément espacées).
    
    Args:
        F: Matrice (n_solutions, n_objectives)
        
    Returns:
        Spacing (0 = parfaitement uniforme)
    """
    if len(F) < 2:
        return 0.0
    
    # Normalise
    F_norm = normalize_objectives(F)
    
    # Calcule les distances à la solution la plus proche
    distances = cdist(F_norm, F_norm, metric='euclidean')
    # Ignore la diagonale (distance à soi-même)
    np.fill_diagonal(distances, np.inf)
    
    # Distance minimale pour chaque solution
    min_distances = distances.min(axis=1)
    
    # Spacing = écart-type des distances minimales
    d_mean = min_distances.mean()
    spacing = np.sqrt(np.sum((min_distances - d_mean) ** 2) / (len(F) - 1))
    
    return spacing

def spread_metric(F: np.ndarray) -> float:
    """
    Calcule la métrique de spread (étendue du front).
    Plus grand = meilleur (front plus étendu).
    
    Args:
        F: Matrice (n_solutions, n_objectives)
        
    Returns:
        Spread (somme des étendues normalisées)
    """
    F_norm = normalize_objectives(F)
    
    # Étendue sur chaque objectif
    spreads = F_norm.max(axis=0) - F_norm.min(axis=0)
    
    return spreads.sum()

def generational_distance(F_approx: np.ndarray, F_true: np.ndarray) -> float:
    """
    Calcule la distance générationnelle entre un front approximé et le vrai front.
    Plus petit = meilleur (plus proche du vrai front).
    
    Args:
        F_approx: Front approximé (n_approx, n_objectives)
        F_true: Vrai front de Pareto (n_true, n_objectives)
        
    Returns:
        Generational distance
    """
    # Normalise ensemble
    F_all = np.vstack([F_approx, F_true])
    F_all_norm = normalize_objectives(F_all)
    
    F_approx_norm = F_all_norm[:len(F_approx)]
    F_true_norm = F_all_norm[len(F_approx):]
    
    # Distance de chaque solution approximée à la plus proche solution vraie
    distances = cdist(F_approx_norm, F_true_norm, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    # GD = moyenne des distances minimales
    gd = min_distances.mean()
    
    return gd

def inverted_generational_distance(F_approx: np.ndarray, F_true: np.ndarray) -> float:
    """
    Calcule l'IGD (distance du vrai front vers le front approximé).
    Plus petit = meilleur (couverture du vrai front).
    
    Args:
        F_approx: Front approximé
        F_true: Vrai front de Pareto
        
    Returns:
        Inverted generational distance
    """
    # C'est la GD dans l'autre sens
    return generational_distance(F_true, F_approx)

def epsilon_indicator(F_approx: np.ndarray, F_true: np.ndarray) -> float:
    """
    Calcule l'indicateur epsilon additif.
    Plus petit = meilleur (facteur d'éloignement minimal).
    
    Args:
        F_approx: Front approximé
        F_true: Vrai front de Pareto
        
    Returns:
        Epsilon indicator
    """
    F_all = np.vstack([F_approx, F_true])
    F_all_norm = normalize_objectives(F_all)
    
    F_approx_norm = F_all_norm[:len(F_approx)]
    F_true_norm = F_all_norm[len(F_approx):]
    
    # Pour chaque point du vrai front, trouve le epsilon minimal
    epsilon_values = []
    
    for true_point in F_true_norm:
        # Distance maximale sur chaque objectif vers les points approximés
        differences = F_approx_norm - true_point
        epsilon_per_approx = differences.max(axis=1)
        epsilon_values.append(epsilon_per_approx.min())
    
    return max(epsilon_values)

def calculate_all_metrics(F: np.ndarray, F_reference: np.ndarray = None) -> dict:
    """
    Calcule toutes les métriques de qualité pour un front de Pareto.
    
    Args:
        F: Front à évaluer (n_solutions, n_objectives)
        F_reference: Front de référence pour GD/IGD (optionnel)
        
    Returns:
        Dictionnaire des métriques
    """
    n_obj = F.shape[1]
    
    metrics = {
        'n_solutions': len(F),
        'spacing': spacing_metric(F),
        'spread': spread_metric(F),
    }
    
    # Hypervolume selon le nombre d'objectifs
    if n_obj == 2:
        metrics['hypervolume'] = hypervolume_2d(F)
    elif n_obj == 3:
        metrics['hypervolume'] = hypervolume_3d(F)
    else:
        metrics['hypervolume'] = None
    
    # Métriques de comparaison si référence fournie
    if F_reference is not None:
        metrics['gd'] = generational_distance(F, F_reference)
        metrics['igd'] = inverted_generational_distance(F, F_reference)
        metrics['epsilon'] = epsilon_indicator(F, F_reference)
    
    return metrics

def compare_fronts(fronts: List[np.ndarray], names: List[str]) -> dict:
    """
    Compare plusieurs fronts de Pareto.
    
    Args:
        fronts: Liste de matrices (n_solutions, n_objectives)
        names: Noms des fronts
        
    Returns:
        DataFrame comparatif
    """
    import pandas as pd
    
    results = []
    
    # Trouve le meilleur front comme référence (celui avec le plus grand hypervolume)
    hvs = []
    for F in fronts:
        if F.shape[1] == 2:
            hvs.append(hypervolume_2d(F))
        else:
            hvs.append(hypervolume_3d(F))
    
    best_idx = np.argmax(hvs)
    F_reference = fronts[best_idx]
    
    # Calcule les métriques pour chaque front
    for name, F in zip(names, fronts):
        metrics = calculate_all_metrics(F, F_reference if F is not F_reference else None)
        metrics['name'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[['name'] + [col for col in df.columns if col != 'name']]
    
    return df

def convergence_plot_data(res_history: List) -> dict:
    """
    Prépare les données pour un graphique de convergence NSGA-II.
    
    Args:
        res_history: Liste des résultats à chaque génération
        
    Returns:
        Dictionnaire avec les séries temporelles
    """
    generations = []
    hvs = []
    spacings = []
    n_sols = []
    
    for gen, res in enumerate(res_history):
        F = res.F
        generations.append(gen)
        n_sols.append(len(F))
        
        if F.shape[1] == 2:
            hvs.append(hypervolume_2d(F))
        else:
            hvs.append(hypervolume_3d(F))
        
        spacings.append(spacing_metric(F))
    
    return {
        'generation': generations,
        'hypervolume': hvs,
        'spacing': spacings,
        'n_solutions': n_sols
    }

# Exemple d'utilisation
if __name__ == "__main__":
    # Génère deux fronts de test
    n = 50
    F1 = np.random.rand(n, 2)
    F2 = np.random.rand(n, 2) * 0.8 + 0.1  # Légèrement différent
    
    print("=== Métriques Front 1 ===")
    metrics1 = calculate_all_metrics(F1)
    for key, val in metrics1.items():
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")
    
    print("\n=== Comparaison des Fronts ===")
    df = compare_fronts([F1, F2], ['Front A', 'Front B'])
    print(df.to_string(index=False))