#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIF72 — EXAMEN PRÉPARATION (Devoir écrit : 2 h, papier, 60 %)
=================================================================
Le DE est SUR PAPIER : il faut savoir faire ces calculs SANS Python.
Chaque bloc ci-dessous est un "drill" : énoncé, méthode à retenir,
corrigé pas-à-pas (à recopier à la main), puis une vérification
Python qui confirme le résultat attendus.

Structure simulée du DE (120 min) :
  Partie A : algèbre linéaire + géométrie        (30 min)
  Partie B : calcul/optimisation + probabilités (30 min)
  Partie C : statistique + tests + régression    (60 min)

Lancer :  python3 code/08_examen_preparation.py
"""
import numpy as np
from scipy import stats

np.set_printoptions(precision=4, suppress=True, linewidth=100)


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ======================================================================
# PARTIE A — ALGÈBRE & GÉOMÉTRIE  (30 min)
# ======================================================================
sep("A1 · Système linéaire par élimination de Gauss (drill papier)")
print("""
Énoncé : Résoudre  { 2x + y - z = 4
                        x + 2y + z = 5
                        x - y + 2z = 1
Méthode papier : éliminer une inconnue à la fois, remonter (substitution).
Corrigé :
  (L1) - 2(L2) : -3y - 3z = -6  ->  y + z = 2           (I)
  (L3) - (L2)  : -3y +  z = -4                    (II)
  z = 2 - y  dans (II) : -3y + (2-y) = -4  ->  -4y = -6  ->  y = 1.5
  z = 2 - 1.5 = 0.5
  x depuis (L2) : x = 5 - 2(1.5) - 0.5 = 1.5
  Solution : x=1.5, y=1.5, z=0.5
""")
A = np.array([[2, 1, -1], [1, 2, 1], [1, -1, 2]])
b = np.array([4, 5, 1])
sol = np.linalg.solve(A, b)
print("Vérification Python :", sol, "-> (1.5, 1.5, 0.5)")
assert np.allclose(sol, [1.5, 1.5, 0.5])


sep("A2 · Déterminant, inverse, matrice transposée (drill papier)")
print("""
Énoncé : M = [[4, 2], [3, 1]]. Calculer det, M^-1, et vérifier M.M^-1 = I.
Méthode : det(M) = 4*1 - 2*3 = -2
         M^-1 = (1/det) * [[ d, -b ],[ -c, a ]] = (-1/2)[[1, -2],[-3, 4]]
Corrigé : M^-1 = [[-0.5, 1], [1.5, -2]]
""")
M = np.array([[4, 2], [3, 1]])
det = np.linalg.det(M)
Minv = np.linalg.inv(M)
print(f"Vérif Python : det={det}  M^-1=\n{Minv}")
assert np.isclose(det, -2)
assert np.allclose(Minv, [[-0.5, 1], [1.5, -2]])
assert np.allclose(M @ Minv, np.eye(2))


sep("A3 · Produit matriciel 2x3 @ 3x2 (drill papier)")
print("""
Énoncé : A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]. Calculer A@B.
Méthode : (i,j) = somme des produits A[i,k]*B[k,j].
Corrigé :
  (1,1)=1*1+2*0+3*1=4   (1,2)=1*0+2*1+3*1=5
  (2,1)=4*1+5*0+6*1=10  (2,2)=4*0+5*1+6*1=11
""")
A3 = np.array([[1, 2, 3], [4, 5, 6]])
B3 = np.array([[1, 0], [0, 1], [1, 1]])
print("Vérif Python :\n", A3 @ B3)
assert np.allclose(A3 @ B3, [[4, 5], [10, 11]])


sep("A4 · Valeurs propres d'une 2x2 (drill papier)")
print("""
Énoncé : P = [[3,1],[1,3]]. Trouver valeurs et vecteurs propres.
Méthode : det(P - λI) = 0  ->  (3-λ)² - 1 = 0  ->  λ² - 6λ + 8 = 0
         Δ = 36-32 = 4  ->  λ = (6±2)/2  =>  λ1=4, λ2=2
Vecteurs propres :
  λ1=4 : (P-4I)v=0 -> [[-1,1],[1,-1]]v=0 -> v1=(1,1)
  λ2=2 : (P-2I)v=0 -> [[1,1],[1,1]]v=0  -> v2=(1,-1)
""")
P4 = np.array([[3, 1], [1, 3]])
e, v = np.linalg.eigh(P4)
print("Vérif Python : valeurs propres =", e)
assert np.allclose(sorted(e), [2, 4])
# vecteurs propres normalisés
assert np.allclose(np.abs(v[:, 0]), [1 / np.sqrt(2), 1 / np.sqrt(2)])


sep("A5 · Normes et angle (drill papier)")
print("""
Énoncé : u=(2,1,2), v=(1,-1,1). Calculer ||u||, u·v, l'angle entre u et v.
Méthode : ||u||=√(4+1+4)=3 ;  u·v=2-1+2=3 ;  cos θ = 3/(||u||·||v||)
         ||v||=√3  ->  cos θ = 3/(3√3) = 1/√3  ->  θ = arccos(0.5774) ≈ 54.74°
""")
u5 = np.array([2., 1, 2]); v5 = np.array([1., -1, 1])
ang5 = np.degrees(np.arccos(u5 @ v5 / (np.linalg.norm(u5) * np.linalg.norm(v5))))
print(f"Vérif Python : angle = {ang5:.2f}°")
assert np.isclose(ang5, np.degrees(np.arccos(1 / np.sqrt(3))))


# ======================================================================
# PARTIE B — CALCUL, OPTIMISATION & PROBABILITÉS  (30 min)
# ======================================================================
sep("B1 · Dérivées et règles (drill papier)")
print("""
Énoncé : f(x) = x⁴ + 3x² - 2x + 5. Trouver f'(x), f'(1), et le point
         d'inflexion ? (f''(x)=0)
Corrigé : f'(x) = 4x³ + 6x - 2
          f'(1)  = 4+6-2 = 8
          f''(x) = 12x² + 6  ->  f''(x) > 0 ∀x  ->  pas d'inflexion
          (f est strictement convexe)
""")
x = np.linspace(-3, 3, 100)
f1 = x**4 + 3*x**2 - 2*x + 5
print(f"Vérif Python : f'(1) = {4*1**3+6*1-2} ; f'' min = {min(12*t**2+6 for t in x)} > 0")
assert 4 + 6 - 2 == 8


sep("B2 · Gradient et point critique (drill papier)")
print("""
Énoncé : g(x,y) = x² + y² - 4x + 2y - 1. Trouver le point critique et
         dire s'il est min/max/selle via le Hessian.
Corrigé : ∇g = (2x-4, 2y+2) = 0  ->  x=2, y=-1
         Hessian = [[2,0],[0,2]] (valeurs propres 2,2 > 0) -> MINIMUM
         g(2,-1) = 4+1-8-2-1 = -6
""")
g2 = lambda x, y: x**2 + y**2 - 4*x + 2*y - 1
print(f"Vérif Python : gradient en (2,-1) = "
      f"({2*2-4}, {2*(-1)+2}) = (0,0) ; g(2,-1)={g2(2,-1)}")
assert g2(2, -1) == -6


sep("B3 · Descente de gradient — pas 1 (drill papier)")
print("""
Énoncé : f(x)=(x-3)², x0=0, α=0.5. Faire 2 pas de GD et donner f après.
Méthode : f'(x)=2(x-3)
          x1 = 0 - 0.5*2(0-3) = 0 + 3 = 3
          x2 = 3 - 0.5*2(3-3) = 3   (on est déjà au minimum)
          f(x1)=0, f(x2)=0
""")
x0, alpha = 0.0, 0.5
x1 = x0 - alpha * 2 * (x0 - 3)
x2 = x1 - alpha * 2 * (x1 - 3)
print(f"Vérif Python : x1={x1}, x2={x2}, f={ (x1-3)**2 }, { (x2-3)**2 }")
assert x1 == 3.0 and x2 == 3.0


sep("B4 · Probabilité bayésienne (drill papier)")
print("""
Énoncé : P(A)=0.4, P(B|A)=0.7, P(B|Aᶜ)=0.3. Calculer P(B) et P(A|B).
Corrigé : P(B) = 0.7*0.4 + 0.3*0.6 = 0.28 + 0.18 = 0.46
          P(A|B) = 0.28/0.46 ≈ 0.6087
""")
pB = 0.7 * 0.4 + 0.3 * 0.6
pA_B = 0.7 * 0.4 / pB
print(f"Vérif Python : P(B)={pB}, P(A|B)={pA_B:.4f}")
assert np.isclose(pB, 0.46) and np.isclose(pA_B, 14 / 23)


sep("B5 · Loi normale (drill papier)")
print("""
Énoncé : X ~ N(μ=60, σ=10). Calculer P(50 < X < 70), P(X > 80), et le
         percentile 95.
Méthode : standardiser z = (x-μ)/σ
          P(50<X<70) = Φ(1) - Φ(-1) = 2Φ(1)-1 ≈ 2(0.8413)-1 = 0.6826
          P(X>80)    = 1-Φ(2) ≈ 1-0.9772 = 0.0228
          P95       = μ + σ·z0.95 = 60 + 10(1.6449) = 76.45
""")
z1 = stats.norm.cdf(1) - stats.norm.cdf(-1)
z2 = 1 - stats.norm.cdf(2)
z3 = 60 + 10 * stats.norm.ppf(0.95)
print(f"Vérif Python : P(50<X<70)={z1:.4f}, P(X>80)={z2:.4f}, "
      f"P95={z3:.3f}")
assert np.isclose(z1, 0.6826, atol=1e-3) and np.isclose(z2, 0.0228, atol=1e-3)


sep("B6 · Loi binomiale (drill papier)")
print("""
Énoncé : n=10 lancers, p=0.5. P(X=5) et P(X≥8).
Méthode : P(X=k) = C(n,k) p^k (1-p)^(n-k)
          P(X=5) = C(10,5)/2^10 = 252/1024 ≈ 0.2461
          P(X≥8) = (C(10,8)+C(10,9)+C(10,10))/1024 = (45+10+1)/1024
                 = 56/1024 ≈ 0.0547
""")
p5 = stats.binom.pmf(5, 10, 0.5)
pge8 = 1 - stats.binom.cdf(7, 10, 0.5)
print(f"Vérif Python : P(X=5)={p5:.4f}, P(X≥8)={pge8:.4f}")
assert np.isclose(p5, 252 / 1024) and np.isclose(pge8, 56 / 1024)


# ======================================================================
# PARTIE C — STATISTIQUE, TESTS, RÉGRESSION  (60 min)
# ======================================================================
sep("C1 · Moyenne, variance, écart-type (drill papier)")
print("""
Énoncé : Echantillon [2, 4, 4, 4, 5, 5, 7, 9] (n=8). Calculer moyenne,
         variance (échantillonnale, n-1), écart-type.
Corrigé : Σ = 40 -> moyenne = 5
          Σ(xi-5)² = 9+1+1+1+0+0+4+16 = 32
          variance = 32/(8-1) = 32/7 ≈ 4.571
          sd = √4.571 ≈ 2.138
""")
xc = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
print(f"Vérif Python : moyenne={xc.mean()}, var={xc.var(ddof=1):.4f}, "
      f"sd={xc.std(ddof=1):.4f}")
assert np.isclose(xc.mean(), 5) and np.isclose(xc.var(ddof=1), 32 / 7)


sep("C2 · Intervalle de confiance (drill papier)")
print("""
Énoncé : n=25, xbar=10, s=2.5. IC95 pour μ (variance inconnue, t).
Méthode : t0.975,24 ≈ 2.064
          IC = xbar ± t·s/√n = 10 ± 2.064·2.5/5 = 10 ± 1.032
          IC95 = [8.968, 11.032]
""")
tcrit = stats.t.ppf(0.975, 24)
ic = (10 - tcrit * 2.5 / 5, 10 + tcrit * 2.5 / 5)
print(f"Vérif Python : t={tcrit:.3f}, IC95=[{ic[0]:.3f},{ic[1]:.3f}]")
assert np.isclose(ic[0], 8.968, atol=0.01) and np.isclose(ic[1], 11.032, atol=0.01)


sep("C3 · Test de deux moyennes (drill papier)")
print("""
Énoncé : G1: n=30, xbar=52, s=5 ; G2: n=30, xbar=55, s=4.
         H0: μ1=μ2 vs H1: μ1≠μ2. Test t à deux échantillons (α=0.05).
Méthode : se = √(s1²/n1 + s2²/n2) = √(25/30 + 16/30) = √(41/30) ≈ 1.168
          t = (55-52)/1.168 ≈ 2.568
          df ≈ 58 -> tcrit ≈ 2.002
          |t| > tcrit -> rejeter H0 (différence significative)
""")
se = np.sqrt(25 / 30 + 16 / 30)
t = 3 / se
print(f"Vérif Python : se={se:.4f}, t={t:.4f}, "
      f"p={2*(1-stats.t.cdf(t, 58)):.5f} -> rejet")
assert t > 2.5 and 2 * (1 - stats.t.cdf(t, 58)) < 0.05


sep("C4 · Régression linéaire (drill papier)")
print("""
Énoncé : Points (0,1),(1,2),(2,5),(3,4),(4,5). Trouver pente β,
         ordonnée α, R².
Corrigé : xbar=2, ybar=3.4
          Sxx = Σ(x-xbar)² = 4+1+0+1+4 = 10
          Sxy = Σ(x-xbar)(y-ybar) = (-2)(-2.4)+(-1)(-1.4)+0+1(0.6)+2(1.6)
                = 4.8+1.4+0+0.6+3.2 = 10.0
          β = Sxy/Sxx = 1.0
          α = ybar - β·xbar = 3.4 - 2 = 1.4
          Prédictions : 1.4, 2.4, 3.4, 4.4, 5.4
          SSE = (1-1.4)²+(2-2.4)²+(5-3.4)²+(4-4.4)²+(5-5.4)²
              = 0.16+0.16+2.56+0.16+0.16 = 3.2
          SST = Σ(y-ybar)² = 5.76+1.96+2.56+0.36+2.56 = 13.2
          R² = 1 - SSE/SST = 1 - 3.2/13.2 ≈ 0.7576
""")
xr = np.array([0, 1, 2, 3, 4], dtype=float)
yr = np.array([1, 2, 5, 4, 5], dtype=float)
xb, yb = xr.mean(), yr.mean()
beta = np.sum((xr - xb) * (yr - yb)) / np.sum((xr - xb) ** 2)
alpha = yb - beta * xb
r2 = 1 - np.sum((yr - (alpha + beta * xr)) ** 2) / np.sum((yr - yb) ** 2)
print(f"Vérif Python : β={beta:.3f}, α={alpha:.3f}, R²={r2:.4f}")
assert np.isclose(beta, 1.0) and np.isclose(alpha, 1.4) and np.isclose(r2, 0.7576, atol=1e-3)


sep("C5 · ANOVA (drill papier)")
print("""
Énoncé : 3 groupes, n=4 chacun. G1=[2,3,2,3], G2=[4,5,4,4], G3=[5,6,5,4].
         H0 : μ1=μ2=μ3. Calculer F et comparer à Fcrit(2,9)=4.256 (α=0.05).
Méthode : moyennes : m1=2.5, m2=4.25, m3=5.0 ; moyenne globale=47/12≈3.9167
          SSB = 4[(2.5-3.9167)² + (4.25-3.9167)² + (5-3.9167)²]
              = 4[2.0069 + 0.1111 + 1.1736] = 4(3.2917) ≈ 13.167
          SSW = Σ interne = G1:1.0 + G2:0.75 + G3:2.0 = 3.75
          F = (SSB/2)/(SSW/9) = 6.583/0.4167 ≈ 15.80
          Fcrit(2,9)=4.256 ;  15.80 > 4.256 -> rejeter H0 (groupes différents)
""")
G1, G2, G3 = np.array([2., 3, 2, 3]), np.array([4., 5, 4, 4]), np.array([5., 6, 5, 4])
f_val, p_val = stats.f_oneway(G1, G2, G3)
print(f"Vérif Python : F={f_val:.3f}, p={p_val:.4f} -> rejet fort")
assert np.isclose(f_val, 15.80, atol=0.1) and p_val < 0.01


sep("C6 · Test du chi-deux (drill papier)")
print("""
Énoncé : Tableau de contingence 2x2 :
                Oui   Non
          Hommes   30    20
          Femmes   20    30
          H0 : sexe indépendant de la réponse.
Méthode : effectifs attendus = (lignetot*coltot)/total
          E[1,1]=50*50/100=25, E[1,2]=25, E[2,1]=25, E[2,2]=25
          chi2 = Σ (obs-att)²/att = 4*(25/25) = 4
          df = (2-1)(2-1) = 1 ; chi2crit(0.05)=3.841
          4 > 3.841 -> rejeter H0 (faiblement lié)
""")
obs = np.array([[30, 20], [20, 30]])
chi2, p, df, _ = stats.chi2_contingency(obs, correction=False)
print(f"Vérif Python : chi2={chi2:.3f}, p={p:.4f} (correction off)")
assert np.isclose(chi2, 4.0) and p < 0.05


# ======================================================================
# SIMULATION BLANCHE — chronomètre
# ======================================================================
sep("SIMULATION BLANCHE (120 min)")
print("""
Partie A (30 min) : 1 système linéaire + 1 déterm/inverse + 1 valeurs propres
Partie B (30 min) : 1 gradient + 1 GD pas à pas + 1 bayes + 1 normale
Partie C (60 min) : 1 IC + 1 test t + 1 régression complète + 1 ANOVA
Conseils examen :
  1. Écrire la méthode AVANT de calculer (barème sur la démarche).
  2. Toujours vérifier (ex : A@x = b ; det(M)M^-1 = I ; R² ∈ [0,1]).
  3. Ne pas sauter d'étapes : chaque ligne de calcul = 1/2 point potentiel.
  4. Si bloqué : poser les équations, c'est déjà 30-40 % du point.
""")

print("\nTous les drills de l'examen sont vérifiés. ✅")
