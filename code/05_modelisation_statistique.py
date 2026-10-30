#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIF72 — T05 · MODÉLISATION STATISTIQUE
========================================
- Distributions d'échantillonnage
- Théorème de la Limite Centrale (TCL)
- Estimation : estimateur, biais, variance
- Intervalles de confiance (moyenne, proportion)
- Régression linéaire et analyse de corrélation

Lancer :  python3 code/05_modelisation_statistique.py
"""
import numpy as np
import pandas as pd
from scipy import stats
import sklearn.linear_model as lm

np.set_printoptions(precision=4, suppress=True, linewidth=100)


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ----------------------------------------------------------------------
# 1) DISTRIBUTIONS D'ÉCHANTILLONNAGE
# ----------------------------------------------------------------------
sep("1. DISTRIBUTIONS D'ÉCHANTILLONNAGE")
# Population : N(mu=10, sigma^2=9). Tirons n=25 -> Xbar ~ N(10, 9/25)
mu, sigma, n = 10.0, 3.0, 25
rng = np.random.default_rng(0)
# 2000 échantillons de taille n, moyenne de chacun
means = []
for _ in range(2000):
    sample = rng.normal(mu, sigma, n)
    means.append(sample.mean())
means = np.array(means)
print(f"Population N({mu},{sigma}) ; échantillons n={n}")
print(f"Moyenne des Xbar = {means.mean():.4f}  (≈ {mu})")
print(f"SD des Xbar      = {means.std(ddof=1):.4f}  (≈ {sigma/np.sqrt(n):.4f})")
assert abs(means.mean() - mu) < 0.1
assert abs(means.std(ddof=1) - sigma / np.sqrt(n)) < 0.1
# Distribution de Xbar est quasi normale même si source non normale (voir TCL)


# ----------------------------------------------------------------------
# 2) THÉORÈME DE LA LIMITE CENTRALE
# ----------------------------------------------------------------------
sep("2. THÉORÈME DE LA LIMITE CENTRALE")
# Population uniforme [0,1] (mu=0.5, sigma^2=1/12). La moyenne de n
# tirages tend vers une normale, même si la source n'est PAS normale.
n = 30
mu_u, var_u = 0.5, 1 / 12
Z = []
for _ in range(5000):
    s = rng.uniform(0, 1, n).mean()
    Z.append((s - mu_u) / np.sqrt(var_u / n))  # standardisation du TIRE
Z = np.array(Z)
# Comparaison avec N(0,1) : moments
print(f"Tirages de la moyenne de {n} uniformes, standardisés")
print(f"  E[Z]  = {Z.mean():.4f}  (attendu 0)")
print(f"  Var[Z]= {Z.var():.4f}  (attendu 1)")
print(f"  kurtosis (excès) = {stats.kurtosis(Z):.4f}  (normale: 0)")
# Test de normalité (Shapiro sur un sous-échantillon)
w, p = stats.shapiro(Z[:500])
print(f"  Shapiro-Wilk p = {p:.3f}  (>{0.05} => compatible normalité)")
assert abs(Z.mean()) < 0.05 and abs(Z.var() - 1) < 0.1


# ----------------------------------------------------------------------
# 3) ESTIMATION — biais & variance d'un estimateur
# ----------------------------------------------------------------------
sep("3. BIAS & VARIANCE D'UN ESTIMATEUR")
# Estimer mu (moyenne) à partir de la moyenne vs de la 1re observation
# (toutes deux sans biais si X~N), mais variances très différentes.
def bias_variance(estimator, n_trials=20000, rng=np.random.default_rng(1)):
    vals = [estimator(rng) for _ in range(n_trials)]
    vals = np.array(vals)
    bias = vals.mean() - mu
    var = vals.var()
    return bias, var, vals


est_mean = lambda r: r.normal(mu, sigma, size=25).mean()
est_first = lambda r: r.normal(mu, sigma, size=1)[0]
bm, vm, _ = bias_variance(est_mean)
bf, vf, _ = bias_variance(est_first)
print(f"Estimateur = moyenne (n=25) : bias={bm:+.4f}, var={vm:.4f}, "
      f"MSE={bm**2+vm:.4f}")
print(f"Estimateur = 1er obs       : bias={bf:+.4f}, var={vf:.4f}, "
      f"MSE={bf**2+vf:.4f}")
print("-> la moyenne est sans biais et bien moins variable (var/sigma^2 ≈ "
      f"{vm/sigma**2:.4f} = 1/n)")
assert abs(bm) < 0.02 and vm < vf


# ----------------------------------------------------------------------
# 4) INTERVALLES DE CONFIANCE (moyenne)
# ----------------------------------------------------------------------
sep("4. INTERVALLES DE CONFIANCE")
# Cas 1 : variance connue -> intervalle par la loi normale
data = rng.normal(mu, sigma, size=60)
xbar = data.mean()
se = sigma / np.sqrt(len(data))
z975 = stats.norm.ppf(0.975)
ic_mu = (xbar - z975 * se, xbar + z975 * se)
print(f"n={len(data)}, xbar={xbar:.3f}, sigma connue={sigma}")
print(f"IC 95% (normale)  : [{ic_mu[0]:.3f}, {ic_mu[1]:.3f}]  contient mu? "
      f"{ic_mu[0] <= mu <= ic_mu[1]}")

# Cas 2 : variance inconnue -> intervalle par t de Student
data2 = rng.normal(mu, sigma, size=60)
xbar2 = data2.mean()
s2 = data2.std(ddof=1)
se2 = s2 / np.sqrt(len(data2))
t975 = stats.t.ppf(0.975, df=len(data2) - 1)
ic_t = (xbar2 - t975 * se2, xbar2 + t975 * se2)
print(f"\nIC 95% (t-Student, s inconnue={s2:.3f}): [{ic_t[0]:.3f}, "
      f"{ic_t[1]:.3f}]  contient mu? {ic_t[0] <= mu <= ic_t[1]}")
print(f"  critique z={z975:.4f} vs t={t975:.4f} (t un peu plus large)")


# ----------------------------------------------------------------------
# 5) RÉGRESSION LINÉAIRE
# ----------------------------------------------------------------------
sep("5. RÉGRESSION LINÉAIRE")
# y = 1.5 x + 2 + bruit
x = rng.uniform(0, 10, 100)
y = 1.5 * x + 2 + rng.normal(0, 1, 100)
reg = lm.LinearRegression().fit(x.reshape(-1, 1), y)
pred = reg.predict(x.reshape(-1, 1))
r2 = reg.score(x.reshape(-1, 1), y)
print(f"pente estimée = {reg.coef_[0]:.3f} (vrai 1.5), "
      f"ordonnée = {reg.intercept_:.3f} (vrai 2)")
print(f"R^2 = {r2:.4f}   (1 = ajustement parfait)")
assert abs(reg.coef_[0] - 1.5) < 0.15 and r2 > 0.7

# Régression multiple (2 prédateurs)
Xm = rng.uniform(-3, 3, (150, 2))
beta_true = np.array([2.0, -1.0])
ym = Xm @ beta_true + 0.5 + rng.normal(0, 0.5, 150)
regM = lm.LinearRegression().fit(Xm, ym)
print(f"\nRégression multiple : coefs = {regM.coef_} (vrais [2,-1]), "
      f"intercepte={regM.intercept_:.3f} (vrai 0.5)")
assert np.allclose(regM.coef_, beta_true, atol=0.15)


# ----------------------------------------------------------------------
# 6) ANALYSE DE CORRÉLATION
# ----------------------------------------------------------------------
sep("6. CORRÉLATION & COEFFICIENT DE DÉTERMINATION")
# Corrélations entre variables
df = pd.DataFrame({
    'A': rng.normal(0, 1, 50),
})
df['B'] = df['A'] * 0.8 + rng.normal(0, 1, 50)   # fortement corrélées
df['C'] = rng.normal(0, 1, 50)                    # indépendantes
corr_AB = np.corrcoef(df['A'], df['B'])[0, 1]
corr_AC = np.corrcoef(df['A'], df['C'])[0, 1]
print(f"corr(A,B) = {corr_AB:.3f}  (r^2={corr_AB**2:.3f})")
print(f"corr(A,C) = {corr_AC:.3f}  (≈0 : indépendantes)")
print("Rappel : |r|≤1 ; r^2 = part de variance expliquée")
assert abs(corr_AB) > 0.5 and abs(corr_AC) < 0.3


# ======================================================================
# EXERCICES CORRIGÉS
# ======================================================================
sep("EXERCICES CORRIGÉS")

# ---- EXERCICE 1 (CORRIGÉ) : loi de Xbar ---------------------------------
# X ~ N(mu=5, sigma=2), n=100. Loi de la moyenne Xbar ?
mu1, sig1, n1 = 5, 2, 100
print(f"\nEx1 : Xbar ~ N({mu1}, {sig1**2/n1:.4f}) ; "
      f"sd(Xbar) = {sig1/np.sqrt(n1):.4f}")
assert np.isclose(sig1 / np.sqrt(n1), 0.2)

# ---- EXERCICE 2 (CORRIGÉ) : IC de moyenne (t) --------------------------
# 10 mesures : [52, 49, 50, 55, 51, 48, 53, 50, 49, 52]. IC95 pour mu ?
m = np.array([52, 49, 50, 55, 51, 48, 53, 50, 49, 52], dtype=float)
mb = m.mean(); ms = m.std(ddof=1); n2 = len(m)
tc = stats.t.ppf(0.975, df=n2 - 1)
ic2 = (mb - tc * ms / np.sqrt(n2), mb + tc * ms / np.sqrt(n2))
print(f"Ex2 : xbar={mb:.2f}, s={ms:.3f}, tc={tc:.3f} -> "
      f"IC95=[{ic2[0]:.3f},{ic2[1]:.3f}]")

# ---- EXERCICE 3 (CORRIGÉ) : P(moyenne > seuil) via TCL -----------------
# Population mu=100, sigma=15, n=9. P(Xbar > 105) ?
mu3, sig3, n3 = 100, 15, 9
se3 = sig3 / np.sqrt(n3)
p3 = 1 - stats.norm.cdf(105, mu3, se3)
print(f"Ex3 : P(Xbar>105) = {p3:.4f}  (z = (105-100)/{se3:.1f} = 1)")
assert np.isclose(p3, 0.1587, atol=1e-3)

# ---- EXERCICE 4 (CORRIGÉ) : régression par moindres carrés à la main ---
# Points (0,1),(1,3),(2,4),(3,6) -> pente et ordonnée ?
Xc = np.array([0, 1, 2, 3]); Yc = np.array([1, 3, 4, 6])
xc, yc = Xc.mean(), Yc.mean()
beta = np.sum((Xc - xc) * (Yc - yc)) / np.sum((Xc - xc) ** 2)
alpha = yc - beta * xc
print(f"Ex4 : pente={beta:.3f}, ordonnée={alpha:.3f}")
assert np.isclose(beta, 1.6) and np.isclose(alpha, 1.1)

# ---- EXERCICE 5 (CORRIGÉ) : IC pour une proportion ---------------------
# n=400, k=280 succès. IC95 pour p ?
n5, k5 = 400, 280
phat = k5 / n5
z5 = stats.norm.ppf(0.975)
ic5 = (phat - z5 * np.sqrt(phat * (1 - phat) / n5),
       phat + z5 * np.sqrt(phat * (1 - phat) / n5))
print(f"Ex5 : p-hat={phat:.3f} -> IC95=[{ic5[0]:.4f},{ic5[1]:.4f}]")

print("\nTous les exercices du T05 sont vérifiés. ✅")
