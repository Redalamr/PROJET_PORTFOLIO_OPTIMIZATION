
"""
Module de chargement et prétraitement des données financières
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Tuple, List
from pathlib import Path
def load_tickers_by_sector(json_path: str = None) -> Dict[str, List[str]]:
    """
    Charge la liste des tickers organisés par secteur.
    """
    # Si aucun chemin n'est fourni, on le construit dynamiquement
    if json_path is None:
        # On récupère le dossier où se trouve CE fichier (src/)
        current_file_dir = Path(__file__).parent
        # On remonte d'un niveau (racine) et on descend dans data/
        json_path = current_file_dir.parent / "data" / "tick.json"
    
    with open(json_path, 'r') as f:
        sectors = json.load(f)
    return sectors

def load_price_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Charge tous les fichiers CSV du dossier raw et les combine.
    
    Args:
        data_dir: Dossier contenant les fichiers CSV par secteur
        
    Returns:
        DataFrame avec Date en index et les tickers en colonnes
    """
    all_dfs = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, file), index_col=0, parse_dates=True)
            all_dfs.append(df)
    
    # Combine tous les DataFrames
    combined = pd.concat(all_dfs, axis=1)
    
    # Trie par date
    combined = combined.sort_index()
    
    return combined

def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """
    Calcule les rendements à partir des prix.
    
    Args:
        prices: DataFrame des prix
        method: 'log' pour rendements logarithmiques, 'simple' pour rendements simples
        
    Returns:
        DataFrame des rendements
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    
    return returns.dropna()

def clean_data(data: pd.DataFrame, max_missing_pct: float = 0.1) -> pd.DataFrame:
    """
    Nettoie les données en supprimant les colonnes avec trop de valeurs manquantes
    et en remplissant les autres.
    
    Args:
        data: DataFrame à nettoyer
        max_missing_pct: Pourcentage maximum de valeurs manquantes autorisé
        
    Returns:
        DataFrame nettoyé
    """
    # Supprime les colonnes avec trop de valeurs manquantes
    missing_pct = data.isnull().sum() / len(data)
    cols_to_keep = missing_pct[missing_pct <= max_missing_pct].index
    data_clean = data[cols_to_keep].copy()
    
    # Rempli les valeurs manquantes restantes par forward fill puis backward fill
    data_clean = data_clean.fillna(method='ffill').fillna(method='bfill')
    
    return data_clean

def prepare_data(start_date: str = "2020-01-01", 
                 end_date: str = "2024-12-31",
                 data_dir: str = "data/raw",
                 tick_json_path: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Pipeline complet de préparation des données.
    
    Args:
        start_date: Date de début
        end_date: Date de fin
        data_dir: Dossier des données brutes
        tick_json_path: Chemin optionnel vers le fichier `data/tick.json` (permet d'être explicite
                        lorsque le notebook est exécuté depuis un dossier différent)
        
    Returns:
        prices: Prix nettoyés
        returns: Rendements
        mu: Vecteur des rendements moyens
        Sigma: Matrice de covariance
        ticker_sectors: Dictionnaire {ticker: secteur}
    """
    # Charge les données
    prices = load_price_data(data_dir)
    
    # Filtre par dates
    prices = prices.loc[start_date:end_date]
    
    # Nettoie
    prices = clean_data(prices)
    
    # Calcule les rendements
    returns = calculate_returns(prices, method='log')
    
    # Annualise les statistiques (252 jours de trading par an)
    mu = returns.mean().values * 252
    Sigma = returns.cov().values * 252
    
    # Crée le mapping ticker -> secteur
    # On permet de passer explicitement le chemin vers `tick.json` (utile dans les notebooks)
    sectors = load_tickers_by_sector(tick_json_path)
    ticker_sectors = {}
    for sector, tickers in sectors.items():
        for ticker in tickers:
            if ticker in prices.columns:
                ticker_sectors[ticker] = sector
    
    return prices, returns, mu, Sigma, ticker_sectors

def save_processed_data(returns: pd.DataFrame, output_path: str = "data/processed/returns.csv"):
    """
    Sauvegarde les rendements traités.
    
    Args:
        returns: DataFrame des rendements
        output_path: Chemin de sortie
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    returns.to_csv(output_path)
    print(f"Rendements sauvegardés dans {output_path}")