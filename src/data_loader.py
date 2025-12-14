"""
Module de chargement et prétraitement des données financières
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Tuple, List, Union # Ajout de Union pour les types optionnels
from pathlib import Path
from logging import getLogger # Bonne pratique pour le logging

# Configuration du logging de base (peut être étendu)
logger = getLogger(__name__)

# Définition de types alias pour la clarté
# Remarque : pd.DataFrame est une classe, pas besoin d'utiliser un alias de type
TickerMap = Dict[str, List[str]]
TickerSectorMap = Dict[str, str]

def load_tickers_by_sector(json_path: Union[str, Path, None] = None) -> TickerMap:
    """
    Charge la liste des tickers organisés par secteur depuis un fichier JSON.

    Args:
        json_path: Chemin vers le fichier JSON. Si None, le chemin 'data/tick.json'
                   est construit dynamiquement par rapport au dossier parent de ce module.

    Returns:
        Dictionnaire où la clé est le nom du secteur et la valeur est une liste de tickers.
    """
    if json_path is None:
        # Utilisation de Path pour construire le chemin de manière portable
        current_file_dir = Path(__file__).resolve().parent
        # Remonter d'un niveau (racine du projet) et descendre dans data/tick.json
        json_path_obj = current_file_dir.parent / "data" / "tick.json"
    else:
        json_path_obj = Path(json_path)
    
    # Vérification d'existence pour une meilleure robustesse
    if not json_path_obj.exists():
        logger.error(f"Le fichier JSON des tickers est introuvable à : {json_path_obj}")
        raise FileNotFoundError(f"Le fichier des tickers est introuvable à {json_path_obj}")

    with open(json_path_obj, 'r', encoding='utf-8') as f: # Ajout de l'encodage
        sectors: TickerMap = json.load(f)
        
    logger.info(f"Chargement réussi des tickers depuis {json_path_obj}.")
    return sectors

def load_price_data(data_dir: Union[str, Path] = "data/raw") -> pd.DataFrame:
    """
    Charge et combine tous les fichiers CSV du dossier spécifié.
    
    Args:
        data_dir: Dossier contenant les fichiers CSV des prix.
        
    Returns:
        DataFrame combiné avec Date en index et les tickers en colonnes.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        logger.error(f"Le répertoire de données est introuvable : {data_path}")
        raise FileNotFoundError(f"Le répertoire de données brutes est introuvable : {data_path}")
        
    all_dfs = []
    
    # Utilisation de Path.glob() qui est plus idiomatique que os.listdir()
    csv_files = list(data_path.glob('*.csv'))
    if not csv_files:
        logger.warning(f"Aucun fichier CSV trouvé dans le répertoire : {data_path}")
        return pd.DataFrame() # Retourne un DataFrame vide si rien n'est trouvé

    for file_path in csv_files:
        try:
            # Enlever le 'os.path.join', Path gère l'objet directement
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            all_dfs.append(df)
            logger.debug(f"Fichier chargé : {file_path.name}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {file_path.name}: {e}")

    # Combine tous les DataFrames (gestion des colonnes/index hétérogènes)
    combined = pd.concat(all_dfs, axis=1, join='outer')
    
    # Le tri par date (index) est une bonne pratique
    combined = combined.sort_index()
    
    logger.info(f"Chargement et combinaison de {len(all_dfs)} fichiers CSV. Forme finale : {combined.shape}")
    return combined

def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """
    Calcule les rendements à partir des prix.
    
    Args:
        prices: DataFrame des prix.
        method: 'log' pour rendements logarithmiques, 'simple' pour rendements simples.
                Gère les majuscules/minuscules pour plus de flexibilité.
        
    Returns:
        DataFrame des rendements.
    """
    method = method.lower()
    
    if prices.empty:
        return pd.DataFrame()

    if method == 'log':
        # np.log(a / b) est équivalent à np.log(a) - np.log(b)
        returns = np.log(prices).diff() 
        logger.debug("Calcul des rendements logarithmiques.")
    elif method == 'simple':
        returns = prices.pct_change()
        logger.debug("Calcul des rendements simples.")
    else:
        logger.warning(f"Méthode de rendement inconnue : {method}. Utilisation par défaut des rendements logarithmiques.")
        returns = np.log(prices).diff()

    # Supprimer la première ligne qui contient toujours des NaN après le calcul des rendements
    return returns.dropna(how='all')

def clean_data(data: pd.DataFrame, max_missing_pct: float = 0.1) -> pd.DataFrame:
    """
    Nettoie les données en supprimant les colonnes avec trop de valeurs manquantes
    et en remplissant les autres.
    
    Args:
        data: DataFrame à nettoyer.
        max_missing_pct: Pourcentage maximum de valeurs manquantes autorisé pour garder une colonne.
        
    Returns:
        DataFrame nettoyé.
    """
    if data.empty:
        return pd.DataFrame()

    # Supprime les colonnes avec trop de valeurs manquantes
    missing_pct = data.isnull().sum() / len(data)
    cols_to_drop = missing_pct[missing_pct > max_missing_pct].index
    
    data_clean = data.drop(columns=cols_to_drop).copy()
    
    logger.info(f"Suppression de {len(cols_to_drop)} colonnes avec > {max_missing_pct*100}% de NaN.")
    
    # Remplissage : forward fill (garde la dernière valeur connue) puis backward fill (remplit le début)
    # L'enchaînement .ffill().bfill() est correct et idiomatique.
    data_clean = data_clean.ffill().bfill() 
    
    # Vérifie s'il reste des NaN (ne devrait pas arriver après ffill/bfill sauf si la colonne est vide)
    if data_clean.isnull().any().any():
         logger.warning("Des NaN subsistent après le nettoyage (cela pourrait indiquer des colonnes entièrement vides ou des problèmes d'indexation).")
         
    return data_clean

def prepare_data(start_date: str = "2020-01-01", 
                 end_date: str = "2024-12-31",
                 data_dir: Union[str, Path] = "data/raw",
                 tick_json_path: Union[str, Path, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, TickerSectorMap]:
    """
    Pipeline complet de préparation des données.
    
    Args:
        start_date: Date de début (incluse) pour le filtrage.
        end_date: Date de fin (incluse) pour le filtrage.
        data_dir: Dossier des données brutes.
        tick_json_path: Chemin optionnel vers le fichier `data/tick.json`.
            
    Returns:
        Tuple contenant :
        - prices: Prix nettoyés (DataFrame)
        - returns: Rendements (DataFrame)
        - mu: Vecteur des rendements moyens annualisés (np.ndarray)
        - Sigma: Matrice de covariance annualisée (np.ndarray)
        - ticker_sectors: Dictionnaire {ticker: secteur} (Dict)
    """
    logger.info(f"Démarrage du pipeline de préparation des données de {start_date} à {end_date}.")
    
    # 1. Charge les données
    prices_raw = load_price_data(data_dir)
    
    if prices_raw.empty:
        logger.error("Aucune donnée de prix chargée. Arrêt du pipeline.")
        return pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([]), {}
    
    # 2. Filtre par dates
    # Assurez-vous que l'index est bien datetime pour le loc
    prices = prices_raw.loc[start_date:end_date].copy()
    logger.info(f"Données filtrées : de {prices.index.min().date()} à {prices.index.max().date()}")
    
    # 3. Nettoie
    prices = clean_data(prices)
    
    # 4. Calcule les rendements
    returns = calculate_returns(prices, method='log')
    
    if returns.empty:
        logger.error("Le DataFrame de rendements est vide après le nettoyage/calcul.")
        return prices, returns, np.array([]), np.array([]), {}
        
    # 5. Annualise les statistiques
    # Il est crucial que `returns` ne contienne plus de NaN pour ces calculs
    trading_days_per_year = 252 
    
    # .values est nécessaire ici car on veut un np.ndarray pour mu et Sigma
    mu = returns.mean().values * trading_days_per_year
    Sigma = returns.cov().values * trading_days_per_year
    
    logger.info(f"Calcul des statistiques annualisées pour {returns.shape[1]} actifs.")

    # 6. Crée le mapping ticker -> secteur
    sectors: TickerMap = load_tickers_by_sector(tick_json_path)
    ticker_sectors: TickerSectorMap = {}
    
    # Les colonnes de `prices` (et `returns`) sont les tickers effectivement gardés
    valid_tickers = set(prices.columns) 

    for sector, tickers in sectors.items():
        for ticker in tickers:
            if ticker in valid_tickers:
                ticker_sectors[ticker] = sector
                
    logger.info(f"Mapping créé pour {len(ticker_sectors)} tickers restants.")
    
    return prices, returns, mu, Sigma, ticker_sectors

def save_processed_data(data: pd.DataFrame, output_path: Union[str, Path] = "data/processed/returns.csv"):
    """
    Sauvegarde le DataFrame traité (généralement les rendements).
    
    Args:
        data: DataFrame à sauvegarder (ex: les rendements).
        output_path: Chemin de sortie (le dossier sera créé s'il n'existe pas).
    """
    output_file = Path(output_path)
    
    # Création du répertoire parent si nécessaire
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(output_file)
    logger.info(f"Données sauvegardées dans {output_file}")
    print(f"✅ Données sauvegardées dans {output_file}") # Affichage pour l'utilisateur