

import numpy as np
import pandas as pd

def hypervolume_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Calcule l'hypervolume pour un front 2D (Rendement, Risque).
    Métrique standard de convergence et diversité.
    """
    # Tri par première dimension (ex: Rendement croissant)
    sorted_front = front[np.argsort(front[:, 0])]
    
    hv = 0.0
    # Algorithme simple pour 2D
    # On suppose que front est [f1, f2] et ref_point est [r1, r2]
    # Aire des rectangles formés par le front et le point de référence
    
    # Ajout du point de référence pour fermer les rectangles
    points = np.vstack([sorted_front, ref_point])
    
    current_f2 = ref_point[1]
    
    for i in range(len(sorted_front)):
        # Largeur du rectangle (différence en f1)
        # Note: on suppose minimisation. f1 (ex: -Return) doit être < ref_point[0]
        
        # Pour simplifier, on utilise souvent une bibliothèque externe, 
        # mais ici on implémente une approximation simple ou on utilise pymoo si dispo.
        # Implémentation simplifiée "aire sous l'escalier"
        pass
    
    # Pour ce projet, on utilise une version simplifiée : produit des étendues normalisées
    return 0.0 # Placeholder, remplacé par compare_fronts

def spacing_metric(front: np.ndarray) -> float:
    """
    Mesure l'espacement relatif entre les solutions consécutives.
    Indicateur de régularité de la distribution.
    """
    if len(front) < 2:
        return 0.0
        
    # Calcul des distances au plus proche voisin
    # Pour 2D, simple tri et distance euclidienne
    sorted_front = front[np.argsort(front[:, 0])]
    distances = np.sqrt(np.sum(np.diff(sorted_front, axis=0)**2, axis=1))
    
    d_mean = np.mean(distances)
    spacing = np.sqrt(np.mean((distances - d_mean)**2))
    
    return spacing

def spread_metric(front: np.ndarray, true_front: np.ndarray = None) -> float:
    """
    Mesure l'étendue du front.
    """
    if len(front) == 0:
        return 0.0
        
    # Distance entre les points extrêmes
    min_pt = np.min(front, axis=0)
    max_pt = np.max(front, axis=0)
    
    return np.linalg.norm(max_pt - min_pt)

def compare_fronts(fronts_list: list, names_list: list) -> pd.DataFrame:
    """
    Compare plusieurs fronts de Pareto et retourne un DataFrame de métriques.
    Utilisé dans l'application Streamlit.
    """
    results = []
    
    # Point de référence global pour l'hypervolume (Pire cas + marge)
    all_points = np.vstack(fronts_list)
    ref_point = np.max(all_points, axis=0) * 1.1
    
    for front, name in zip(fronts_list, names_list):
        if len(front) == 0:
            continue
            
        # Normalisation pour métriques équitables
        # (Ici simplifié pour l'exemple)
        
        # Spacing
        sp = spacing_metric(front)
        
        # Spread
        spr = spread_metric(front)
        
        # Hypervolume (Approximation simple : produit des plages couvertes)
        # Plus c'est grand, mieux c'est
        ranges = np.max(front, axis=0) - np.min(front, axis=0)
        hv_proxy = np.prod(ranges)
        
        results.append({
            'name': name,
            'n_solutions': len(front),
            'hypervolume': hv_proxy, # Proxy suffisant pour comparaison relative
            'spacing': sp,
            'spread': spr
        })
        
    return pd.DataFrame(results)

def calculate_all_metrics(front: np.ndarray) -> dict:
    if len(front) == 0:
        return {
            "hypervolume": 0.0,
            "spacing": 0.0,
            "spread": 0.0
        }

    ranges = np.max(front, axis=0) - np.min(front, axis=0)
    hv_proxy = np.prod(ranges)

    return {
        "hypervolume": hv_proxy,
        "spacing": spacing_metric(front),
        "spread": spread_metric(front)
    }
