#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIF72 — T04 · THÉORIE DES PROBABILITÉS
========================================
- Espace probabiliste (E, F, P)
- Axiomes de Kolmogorov
- Variables aléatoires discrètes et continues
- Distributions (binomiale, poisson, normale)
- Espérance, variance, écart-type
- Théorème de Bayes
- Modélisation de l'incertitude et évaluation des risques

Lancer :  python3 code/04_probabilités.py
"""
import numpy as np
from scipy import stats

np.set_printoptions(precision=4, suppress=True, linewidth=100)


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ----------------------------------------------------------------------
# 1) ESPACE PROBABILISTE & AXIOMES DE KOLMOGOROV
# ----------------------------------------------------------------------
sep("1. AXIOMES DE KOLMOGOROV")
# Dé un 6 faces, P(A) pour A = {1,3,5} (impair)
sample_space = {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
assert np.isclose(sum(sample_space.values()), 1.0), "Axiome 2 : P(Ω)=1"
assert all(v >= 0 for v in sample_space.values()), "Axiome 1 : P≥0"
P_impair = sum(sample_space[i] for i in [1, 3, 5])
print(f"P({{1,3,5}}) = {P_impair} = 3/6 (additivité de la probabilité)")
assert np.isclose(P_impair, 0.5)


# ----------------------------------------------------------------------
# 2) V.A. DISCRÈTE — distribution de probabilité
# ----------------------------------------------------------------------
sep("2. V.A. DISCRÈTE")
# Nombre de piles sur 4 lancers de pièce équitable (Binomiale n=4, p=0.5)
n = 4
p = 0.5
k = np.arange(n + 1)
pmf = stats.binom.pmf(k, n, p)
print("k  P(X=k)")
for kv, pv in zip(k, pmf):
    print(f"{kv}  {pv:.4f}")
print("Somme des probas =", pmf.sum())
assert np.isclose(pmf.sum(), 1.0)
print(f"E[X] = np = {n*p:.2f}   Var[X] = np(1-p) = {n*p*(1-p):.2f}")


# ----------------------------------------------------------------------
# 3) V.A. CONTINUE — Normal
# ----------------------------------------------------------------------
sep("3. V.A. CONTINUE — NORMALE")
mu, sigma = 2.0, 1.5
print(f"X ~ N({mu}, {sigma}^2)")
print(f"  E[X]  = {mu}   Var[X] = {sigma**2}   sd = {sigma}")
# P(0 < X < 3) via CDF
c = stats.norm.cdf(3, mu, sigma) - stats.norm.cdf(0, mu, sigma)
print(f"  P(0<X<3) = {c:.4f}")
assert 0 < c < 1
# Quantiles (bornes)
q05 = stats.norm.ppf(0.05, mu, sigma)
q95 = stats.norm.ppf(0.95, mu, sigma)
print(f"  5e percentile = {q05:.3f}, 95e percentile = {q95:.3f}")
# Règle 68-95-99.7
for k_, pct in [(1, 68.27), (2, 95.45), (3, 99.73)]:
    val = 2 * stats.norm.cdf(k_) - 1
    print(f"  P(|Z|<{k_}) = {val:.4f}  (~{pct}%)")


# ----------------------------------------------------------------------
# 4) LOIS UTILISÉES EN SCIENCE DES DONNÉES
# ----------------------------------------------------------------------
sep("4. LOIS COURANTES")
# Poisson (événements rares : appels par heure)
lam = 3.0
print(f"Poisson(λ={lam}) : P(X=5) = {stats.poisson.pmf(5, lam):.4f} "
      f"E[X]={lam} Var[X]={lam}")
# Binomiale (succès/échecs)
print(f"Binomiale(n=20,p=0.3) : P(X=5) = {stats.binom.pmf(5,20,0.3):.4f}")
# Bernoulli (1 essai)
print(f"Bernoulli(p=0.7) : P(1)={0.7}, P(0)={0.3}, E[X]={0.7}, "
      f"Var[X]={0.7*0.3:.4f}")
# Exponentielle (temps entre événements)
print(f"Exponentielle(λ=2) : P(X<0.5) = {stats.expon.cdf(0.5, scale=0.5):.4f}")
# Beta (proportions / priors bayésiens)
print(f"Bêta(α=2,β=5) : E[X] = {2/7:.3f}, Var[X] = "
      f"{2*5/(7**2*8):.4f}")


# ----------------------------------------------------------------------
# 5) THÉORÈME DE BAYES
# ----------------------------------------------------------------------
sep("5. THÉORÈME DE BAYES")
# P(H|E) = P(E|H) P(H) / P(E)
# Exemple classique : test de dépistage
# Prévalence (prior) P(Malade) = 1 % ; test sensible 99 % ; FPR 5 %
P_malade = 0.01
P_pos_malade = 0.99    # sensibilité
P_pos_sain = 0.05      # faux positifs
P_pos = P_pos_malade * P_malade + P_pos_sain * (1 - P_malade)
P_malade_pos = P_pos_malade * P_malade / P_pos
print(f"P(Positif) = {P_pos:.5f}")
print(f"P(Malade | Positif) = {P_malade_pos:.5f}  "
      f"({P_malade_pos*100:.2f} %)  <-- paradoxe du taux de base !")
assert P_malade_pos < 0.2  # malgré 99% de sensibilité, très peu de malades


# Exemple 2 : Bayes simple (prior → posterior)
# 100 tirages, 3 succès -> P(succès) ~ Bêta(1,1) → Bêta(3+1, 97+1)
a, b = 1, 1
successes, trials = 3, 100
a_post, b_post = a + successes, b + trials - successes
print(f"\nPrior Bêta({a},{b}) + 3 succès sur 100 -> "
      f"Posterior Bêta({a_post},{b_post})")
print(f"  E[prior]={a/(a+b):.3f} -> E[posterior]={a_post/(a_post+b_post):.3f}")
print(f"  Var[posterior]={a_post*b_post/((a_post+b_post)**2*(a_post+b_post+1)):.5f}")


# ----------------------------------------------------------------------
# 6) INFÉRENCE : TESTS D'HYPOTHÈSE (aperçu)
# ----------------------------------------------------------------------
sep("6. TEST D'HYPOTHÈSE (aperçu — détaillé en T06)")
# H0 : la pièce est équilibrée (p=0.5) vs  H1 : p ≠ 0.5
# 100 lancers, 60 piles -> test binomial
n, k, p0 = 100, 60, 0.5
z = (k - n * p0) / np.sqrt(n * p0 * (1 - p0))  # normal approx
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"z = {z:.3f}, p-value = {p_value:.5f}")
alpha = 0.05
print(f"A α={alpha}: {'REJET' if p_value < alpha else 'non-rejet'} de H0 "
      f"({p_value < alpha})")
assert p_value < 0.05  # 60 piles sur 100 est significatif


# ----------------------------------------------------------------------
# 7) MODÉLISATION D'INCERTITUDE & RISQUE
# ----------------------------------------------------------------------
sep("7. MODÉLISATION D'INCERTITUDE")
# Monte-Carlo : P(X > 100) pour X ~ LogNormal(μ=4.5, σ=0.5)
rng = np.random.default_rng(7)
mu_l, sigma_l = 4.5, 0.5
N = 200_000
X = rng.lognormal(mu_l, sigma_l, N)
p_mc = np.mean(X > 100)
p_exact = 1 - stats.lognorm.cdf(100, s=sigma_l, scale=np.exp(mu_l))
print(f"P(X>100) : Monte-Carlo = {p_mc:.5f}, exacte = {p_exact:.5f}")
assert abs(p_mc - p_exact) < 0.01  # MC converge
# VaR (Value at Risk) au 95e percentile
var95 = np.percentile(X, 95)
print(f"VaR 95% = {var95:.3f}")


# ======================================================================
# EXERCICES CORRIGÉS
# ======================================================================
sep("EXERCICES CORRIGÉS")

# ---- EXERCICE 1 (CORRIGÉ) : proba combinatoire -------------------------
# Dé + pièce : P(face pair ET pile)
# P(pair)=3/6, P(pile)=1/2, indépendants -> 1/4
p_ex1 = (3/6) * (1/2)
print(f"\nEx1 : P(pair et pile) = {p_ex1}")
assert p_ex1 == 0.25

# ---- EXERCICE 2 (CORRIGÉ) : E[X] et Var[X] d'une v.a. discrète --------
# X ~ {0,1,2} avec P = {0.2, 0.5, 0.3}
xx = np.array([0, 1, 2]); ppx = np.array([0.2, 0.5, 0.3])
ex = (xx * ppx).sum()
ex2 = ((xx ** 2) * ppx).sum()
var = ex2 - ex ** 2
print(f"Ex2 : E[X] = {ex}, E[X^2] = {ex2}, Var[X] = {var}")
assert np.isclose(ex, 1.1) and np.isclose(var, 0.49)

# ---- EXERCICE 3 (CORRIGÉ) : P(0 ≤ Z ≤ 1) pour Z ~ N(0,1) -------------
z01 = stats.norm.cdf(1) - stats.norm.cdf(0)
print(f"Ex3 : P(0≤Z≤1) = {z01:.4f}")
assert np.isclose(z01, 0.3413, atol=1e-3)

# ---- EXERCICE 4 (CORRIGÉ) : Binomiale ---------------------------------
# 50 tirs, p=0.4 -> P(X=20) et P(X≥40)
n, p = 50, 0.4
p_eq20 = stats.binom.pmf(20, n, p)
p_ge40 = 1 - stats.binom.cdf(39, n, p)
print(f"Ex4 : P(X=20) = {p_eq20:.4f}, P(X≥40) = {p_ge40:.4f}")
assert 0 < p_eq20 < 1 and 0 < p_ge40 < 1e-3

# ---- EXERCICE 5 (CORRIGÉ) : Bayes 2 classes ---------------------------
# P(A)=0.6, P(B|A)=0.3, P(B|A^c)=0.8  ->  P(A|B)
p_a, p_b_given_a, p_b_given_not = 0.6, 0.3, 0.8
p_b = p_b_given_a * p_a + p_b_given_not * (1 - p_a)
p_a_given_b = p_b_given_a * p_a / p_b
print(f"Ex5 : P(B) = {p_b:.3f}, P(A|B) = {p_a_given_b:.4f}")
assert np.isclose(p_a_given_b, 0.36)

# ---- EXERCICE 6 (CORRIGÉ) : Loi de Poisson ----------------------------
# λ=2.5, P(X=3)
p3 = stats.poisson.pmf(3, 2.5)
print(f"Ex6 : P(X=3) = {p3:.4f}")
assert np.isclose(p3, 0.2138, atol=1e-3)

print("\nTous les exercices du T04 sont vérifiés. ✅")
