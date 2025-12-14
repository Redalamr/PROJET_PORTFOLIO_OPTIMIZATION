"""
Script pour générer tous les résultats du Niveau 3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Imports
import sys
sys.path.insert(0, 'src')

from data_loader import prepare_data
from pareto_metrics import compare_fronts
from robustness import bootstrap_resampling, calculate_var_cvar
from limits_analysis import test_normality, correlation_stability, concentration_analysis
from optimizers.classic import (
    generate_efficient_frontier_scalarization,
    find_tangency_portfolio
)
from optimizers.genetic import optimize_nsga2_biobjective, extract_pareto_front

# Crée dossier results
Path("results").mkdir(exist_ok=True)

print("Chargement des données...")
prices, returns, mu, Sigma, ticker_sectors = prepare_data()
tickers = list(prices.columns)

# === 1. COMPARAISON QUANTITATIVE ===
print("\n1. Comparaison des méthodes...")

_, rets_scal, risks_scal = generate_efficient_frontier_scalarization(mu, Sigma, n_points=50)
F_scal = np.column_stack([-rets_scal, risks_scal])

res_nsga = optimize_nsga2_biobjective(mu, Sigma, K=None, pop_size=100, n_gen=100)
_, F_nsga = extract_pareto_front(res_nsga)

metrics_df = compare_fronts([F_scal, F_nsga], ['Scalarisation', 'NSGA-II'])
metrics_df.to_csv('results/comparison_metrics.csv', index=False)

plt.figure(figsize=(10, 6))
plt.scatter(np.sqrt(risks_scal)*100, rets_scal*100, label='Scalarisation', s=50, alpha=0.7)
plt.scatter(np.sqrt(F_nsga[:, 1])*100, -F_nsga[:, 0]*100, label='NSGA-II', s=50, alpha=0.7, marker='x')
plt.xlabel('Volatilité (%)')
plt.ylabel('Rendement (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Comparaison des Fronts de Pareto')
plt.savefig('results/comparison_fronts.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Sauvegardé: comparison_metrics.csv, comparison_fronts.png")

# === 2. TESTS DE NORMALITÉ ===
print("\n2. Tests de normalité...")

norm_results = test_normality(returns)
norm_results.to_csv('results/normality_tests.csv', index=False)

n_reject = norm_results['JB_reject'].sum()
print(f"   ℹ️  {n_reject}/{len(returns.columns)} actifs rejettent la normalité")
print(f"   ✅ Sauvegardé: normality_tests.csv")

# === 3. STABILITÉ DES CORRÉLATIONS ===
print("\n3. Stabilité des corrélations...")

corr_stab = correlation_stability(returns, n_periods=5)
print(f"   ℹ️  Instabilité moyenne: {corr_stab['mean_stability']:.4f}")

# === 4. BOOTSTRAP ===
print("\n4. Bootstrap (50 échantillons)...")

bootstrap_samples = bootstrap_resampling(returns, n_samples=50)

from optimizers.classic import find_tangency_portfolio
from robustness import optimize_with_bootstrap

w_mean, w_std = optimize_with_bootstrap(
    mu, Sigma, bootstrap_samples,
    optimizer_func=find_tangency_portfolio, rf=0.02
)

top_10 = np.argsort(w_mean)[-10:]
bootstrap_df = pd.DataFrame({
    'Ticker': [tickers[i] for i in top_10],
    'Poids_Moyen (%)': [w_mean[i]*100 for i in top_10],
    'Ecart_type (%)': [w_std[i]*100 for i in top_10],
    'IC_Inf (%)': [(w_mean[i] - 1.96*w_std[i])*100 for i in top_10],
    'IC_Sup (%)': [(w_mean[i] + 1.96*w_std[i])*100 for i in top_10]
})
bootstrap_df.to_csv('results/bootstrap_weights.csv', index=False)

print(f"   ✅ Sauvegardé: bootstrap_weights.csv")

# === 5. CONCENTRATION ===
print("\n5. Analyse de concentration...")

w_tangent = find_tangency_portfolio(mu, Sigma, rf=0.02)
conc = concentration_analysis(w_tangent, tickers, ticker_sectors)

conc_df = pd.DataFrame({
    'Métrique': [
        'Herfindahl Actifs',
        'N Effectif Actifs',
        'Herfindahl Secteurs',
        'N Effectif Secteurs',
        'Top 5 Concentration',
        'Top 10 Concentration'
    ],
    'Valeur': [
        f"{conc['herfindahl_assets']:.4f}",
        f"{conc['effective_n_assets']:.1f}",
        f"{conc['herfindahl_sectors']:.4f}",
        f"{conc['effective_n_sectors']:.1f}",
        f"{conc['top5_concentration']*100:.1f}%",
        f"{conc['top10_concentration']*100:.1f}%"
    ]
})
conc_df.to_csv('results/concentration_analysis.csv', index=False)

print(f"   ℹ️  Top 5 Concentration: {conc['top5_concentration']*100:.1f}%")
print(f"   ✅ Sauvegardé: concentration_analysis.csv")

# === 6. VAR / CVAR ===
print("\n6. VaR et CVaR...")

var, cvar = calculate_var_cvar(returns, w_tangent, confidence=0.95)
print(f"   ℹ️  VaR (95%): {var:.2f}%")
print(f"   ℹ️  CVaR (95%): {cvar:.2f}%")

risk_df = pd.DataFrame({
    'Métrique': ['VaR 95%', 'CVaR 95%', 'Volatilité (Markowitz)'],
    'Valeur (%)': [var, cvar, np.sqrt(w_tangent @ Sigma @ w_tangent)*100]
})
risk_df.to_csv('results/risk_metrics.csv', index=False)

print(f"   ✅ Sauvegardé: risk_metrics.csv")

print("\n" + "="*60)
print("✅ TOUS LES RÉSULTATS ONT ÉTÉ GÉNÉRÉS DANS results/")
print("="*60)
print("\nFichiers créés:")
print("  - comparison_metrics.csv")
print("  - comparison_fronts.png")
print("  - normality_tests.csv")
print("  - bootstrap_weights.csv")
print("  - concentration_analysis.csv")
print("  - risk_metrics.csv")