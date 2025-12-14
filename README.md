# 📈 Projet d'Optimisation de Portefeuille Multi-Critère

## 🎯 Objectif

Ce projet implémente une solution complète d'optimisation de portefeuille financier multi-critère, structurée en trois niveaux de complexité croissante :

- **Niveau 1** : Modèle classique bi-objectif de Markowitz (rendement vs risque)
- **Niveau 2** : Optimisation tri-objectif avec contraintes opérationnelles (cardinalité, coûts de transaction)
- **Niveau 3** : Application interactive Streamlit pour la visualisation et la sélection de portefeuilles

## 📁 Structure du Projet

```
PROJET_PORTFOLIO_OPTIMIZATION/
│
├── data/
│   ├── raw/                    # Données de prix téléchargées
│   ├── processed/              # Rendements calculés
│   └── tick.json               # Configuration des tickers par secteur
│
├── notebooks/                  # Analyses exploratoires (optionnel)
│   ├── 01_data_analysis.ipynb
│   ├── 02_markowitz_dev.ipynb
│   └── 03_nsga2_dev.ipynb
│
├── src/                        # Code source principal
│   ├── __init__.py
│   ├── data_loader.py          # Chargement et prétraitement
│   ├── financial_metrics.py   # Calculs financiers
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── classic.py          # Markowitz (Niveau 1)
│   │   └── genetic.py          # NSGA-II (Niveau 2)
│   └── utils.py
│
├── app.py                      # Application Streamlit (Niveau 3)
├── download.py                 # Script de téléchargement des données
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

Les bibliothèques principales sont :
- **Données** : `numpy`, `pandas`, `yfinance`
- **Optimisation** : `scipy`, `cvxpy`, `pymoo`
- **Visualisation** : `matplotlib`, `seaborn`, `plotly`, `streamlit`

## 📊 Utilisation

### Étape 1 : Téléchargement des Données

Le fichier `download.py` télécharge les historiques de prix depuis Yahoo Finance :

```bash
python download.py
```

Ce script :
- Lit la liste des tickers depuis `data/tick.json`
- Télécharge les prix ajustés de 2020 à 2024
- Sauvegarde les données par secteur dans `data/raw/`

### Étape 2 : Lancement de l'Application

```bash
streamlit run app.py
```

L'application se lance dans votre navigateur à l'adresse `http://localhost:8501`

## 🔬 Fonctionnalités

### Niveau 1 : Markowitz Classique

**Problème d'optimisation :**
```
min  {-w^T μ, w^T Σ w}
s.t. Σ w_i = 1
     w_i ≥ 0
```

**Méthodes disponibles :**
- Scalarisation par somme pondérée
- Méthode epsilon-contrainte
- Identification du portefeuille tangent (Sharpe maximum)

**Visualisations :**
- Frontière efficace 2D (rendement vs risque)
- Composition du portefeuille optimal
- Répartition sectorielle

### Niveau 2 : Contraintes Opérationnelles

**Problème d'optimisation :**
```
min  {-w^T μ, w^T Σ w, Σ|w_i - w_t,i|}
s.t. Σ w_i = 1
     w_i ≥ 0
     Σ I(w_i > δ) = K  (contrainte de cardinalité)
```

**Algorithme :** NSGA-II (Non-dominated Sorting Genetic Algorithm II)

**Paramètres ajustables :**
- Cardinalité K (nombre d'actifs dans le portefeuille)
- Coût proportionnel de transaction
- Taille de la population
- Nombre de générations

**Visualisations :**
- Front de Pareto 3D (rendement, risque, coûts)
- Projections 2D du front
- Sélection interactive d'un portefeuille selon des critères

### Niveau 3 : Comparaison des Méthodes

Compare les résultats de :
- Markowitz classique (scalarisation)
- NSGA-II bi-objectif

Permet d'analyser :
- La qualité des fronts de Pareto
- La diversité des solutions
- Les temps de calcul

## 📖 Formulation Mathématique

### Fonctions Objectifs

1. **Rendement** (à maximiser) :
   ```
   f₁(w) = -w^T μ
   ```
   où μ est le vecteur des rendements moyens annualisés

2. **Risque** (à minimiser) :
   ```
   f₂(w) = w^T Σ w
   ```
   où Σ est la matrice de covariance annualisée

3. **Coûts de transaction** (à minimiser) :
   ```
   f₃(w) = c_prop × Σ|w_i - w_t,i|
   ```
   où w_t est le portefeuille actuel et c_prop le coût proportionnel

### Contraintes

**Contraintes de base :**
- Investissement complet : Σ w_i = 1
- Pas de vente à découvert : w_i ≥ 0

**Contraintes opérationnelles (Niveau 2) :**
- Cardinalité : Σ I(w_i > δ) = K
  (exactement K actifs avec un poids significatif)

## 🎓 Concepts Clés

### Dominance de Pareto

Une solution A domine une solution B si :
- A est au moins aussi bonne que B sur tous les objectifs
- A est strictement meilleure que B sur au moins un objectif

### Front de Pareto

Ensemble des solutions non-dominées. Aucune solution du front ne peut être améliorée sur un objectif sans dégrader au moins un autre objectif.

### NSGA-II

Algorithme génétique multi-objectif qui :
1. Génère une population de solutions
2. Évalue les objectifs et contraintes
3. Classe les solutions par rang de non-dominance
4. Sélectionne, croise et mute pour créer la génération suivante
5. Converge vers le front de Pareto

### Sharpe Ratio

Mesure du rendement ajusté au risque :
```
Sharpe = (r_p - r_f) / σ_p
```
où r_p est le rendement du portefeuille, r_f le taux sans risque, et σ_p la volatilité.

## 📝 Livrables du Projet

1. **Rapport (5-8 pages)** :
   - Présentation de la méthode
   - Formalisation mathématique
   - Comparaison des approches
   - Limites et perspectives

2. **Code Python** :
   - Structure modulaire et réutilisable
   - Documentation inline
   - Code versionné sur GitHub

3. **Application Streamlit** :
   - Interface interactive
   - Visualisations dynamiques
   - Sélection de portefeuilles selon des critères

4. **Présentation orale (15 min)** :
   - Démonstration de l'application
   - Explication des choix méthodologiques
   - Discussion des résultats

## ⚠️ Limites et Perspectives

### Limites Théoriques
- Hypothèse de normalité des rendements
- Stationnarité des statistiques (μ, Σ)
- Absence de contraintes de liquidité

### Limites Statistiques
- Incertitude des estimateurs (μ, Σ)
- Sensibilité aux données historiques
- Risque de surajustement

### Limites Computationnelles
- Temps de calcul pour NSGA-II avec grandes populations
- Convergence non garantie vers l'optimal global
- Trade-off précision/rapidité

### Perspectives d'Amélioration
- Intégration de modèles de robustesse (Black-Litterman, rééchantillonnage)
- Ajout de contraintes ESG
- Backtesting sur données out-of-sample
- Optimisation dynamique (réallocation périodique)

## 🤝 Contribution

Projet réalisé dans le cadre du cours d'optimisation multi-critère.

**Auteurs :** [Vos noms]  
**Date :** Décembre 2025  
**Institution :** [Votre institution]

## 📧 Contact

Pour toute question : fabien.lionti@gmail.com

## 📄 Licence

Ce projet est réalisé dans un cadre académique.

---

**Bon courage pour votre projet ! 🚀**