#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insère les sections manquantes (référencées au livre MML de Deisenroth et al.)
dans les notebooks EXISTANTS :
  01 -> eigendecomposition, Cholesky, PCA latente
  03 -> Jacobien, backpropagation, Hessian/Taylor, Lagrange
  04 -> famille exponentielle, changement de variable
  05 -> régression linéaire bayésienne
"""
import nbformat as nbf
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(BASE, "notebooks")

def md(t):  return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)

# ======================================================================
# 01 — ALGÈBRE LINÉAIRE
# ======================================================================
new01 = [
md("""## 8 · Eigendecomposition & diagonalisation *(MML ch. 4.4)*

Si `A` possède `n` vecteurs propres linéairement indépendants, alors
`A = P Λ P⁻¹` : c'est **diagonaliser** A. Conséquences majeures :
- les puissances `Aⁿ = P Λⁿ P⁻¹` se calculent en une multiplication diagonale
- les exponentielles de matrices `eᴬ = P e^Λ P⁻¹` (équations différentielles, noyaux)
- pour A **symétrique**, P est orthogonale : `P⁻¹ = Pᵀ` (théorème spectral, T01 §5)

**Limite** : une matrice peut ne pas être diagonalisable (valeur propre
multiple mais vecteur propre unique) — c'est le cas des matrices
**nilpotentes** non nulles.""", ),
code("""A = np.array([[4., -2.], [1., 1.]])
eigs, P = np.linalg.eig(A)
Lam = np.diag(eigs)
print("Valeurs propres :", eigs)
print("Reconstruction P @ Λ @ P⁻¹ =\\n", P @ Lam @ np.linalg.inv(P))
assert np.allclose(P @ Lam @ np.linalg.inv(P), A)

# Puissance de matrice : A^15 par diagonalisation (vs itératif)
n = 15
An_ref = np.linalg.matrix_power(A, n)
An_fast = P @ np.diag(eigs ** n) @ np.linalg.inv(P)
print(f"A^{n} identique par diagonalisation :", np.allclose(An_ref, An_fast))
assert np.allclose(An_ref, An_fast)

# Contre-exemple : B = [[0,1],[0,0]] (nilpotente, B² = 0 mais B ≠ 0)
B = np.array([[0., 1.], [0., 0.]])
eb, Pb = np.linalg.eig(B)
print("\\nB nilpotente : valeur propre double =", eb,
      ", rang(P) =", np.linalg.matrix_rank(Pb), "-> NON diagonalisable")
assert np.linalg.matrix_rank(Pb) < 2"""),
md("""## 9 · Décomposition de Cholesky *(MML ch. 4.3)*

Pour toute matrice **symétrique définie positive (SPD)** `C`, il existe
une triangulaire inférieure `L` telle que `C = L Lᵀ`.

**Pourquoi on l'utilise :**
- résoudre `Cx = b` avec une matrice SPD : 2× moins coûteux qu'un LU
- **échantillonner un Gaussien corrélé** : si `z ~ N(0, I)` alors
  `x = L z` a covariance `E[xxᵀ] = L Lᵀ = C`
- base de l'algorithme de Kalman, MCMC, et préconditionnement""", ),
code("""C = np.array([[4., 2.], [2., 3.]])
L = np.linalg.cholesky(C)
print("L =\\n", L)
print("L @ L.T = C ?", np.allclose(L @ L.T, C))
assert np.allclose(L @ L.T, C)

# Échantillonner N(0, C) : x = L z
rng = np.random.default_rng(0)
z = rng.standard_normal((50000, 2))
x = z @ L.T          # Cov(x) = L Lᵀ = C
cov_est = np.cov(x.T)
print("Covariance estimée ≈ C :", np.allclose(cov_est, C, atol=0.1))
assert np.allclose(cov_est, C, atol=0.1)"""),
md("""## 10 · PCA — perspective variable latente *(MML ch. 10.7)*

Le PCA peut se lire comme un **modèle génératif** à `k` facteurs latents :
```
x = W h + ε ,   h ~ N(0, Iₖ) ,   ε ~ N(0, σ² I)
```
- `h` : facteurs latents inobservés (dimension `k` < `n`)
- `W` : projection vers l'espace observé
- L'estimation par **maximum de vraisemblance** de `(W, σ²)` revient
  (à une rotation près) aux composantes principales du PCA.

C'est le pont vers le **factor analysis** et les GMM (T07 / MML ch. 11).""", ),
code("""rng = np.random.default_rng(42)
W_true = np.array([[3.0, 0.5], [0.2, 1.0], [2.5, -0.3]])  # 3 obs, 2 latents
m = 500
h = rng.normal(0, 1, (m, 2))
noise_sd = 0.3
X = h @ W_true.T + noise_sd * rng.normal(0, 1, (m, 3))
Xc = X - X.mean(axis=0)

pca = PCA(n_components=2).fit(Xc)
print(f"Variance expliquée par 2 facteurs : {pca.explained_variance_ratio_.sum()*100:.1f}%")
scores = pca.transform(Xc)
rec = pca.inverse_transform(scores) + X.mean(axis=0)
err = np.linalg.norm(X - rec) / np.linalg.norm(Xc)
print(f"Erreur de reconstruction relative = {err:.4f} (petite -> les 2 latents suffisent)")
assert pca.explained_variance_ratio_.sum() > 0.95
assert err < 0.02"""),
]

# ======================================================================
# 03 — CALCUL & OPTIMISATION
# ======================================================================
new03 = [
md("""## 7 · Jacobien & gradients de fonctions vectorielles *(MML ch. 5.3-5.4)*

Pour `f : ℝⁿ → ℝᵐ`, le **Jacobian** est la matrice `m×n` des dérivées
partielles : `Jᵢⱼ = ∂fᵢ/∂xⱼ`. C'est le **linéarisation de premier ordre**
de `f` (analogue du gradient pour les fonctions scalaires).

Règles essentielles :
- **chaîne** : `J_{g∘f} = J_g · J_f` (le backbone de la backpropagation)
- le gradient d'une fonction scalaire `ℒ(x)` **est** le Jacobien
  transposé vu comme un vecteur : `∇ℒ = J_ℒᵀ`
- en ML on ne calcule presque jamais `J` entier : on calcule des
  **produits vecteur-Jacobien (VJP)** — c'est exactement ce que fait
  la rétropropagation.

*À l'examen* : écrire `J` à la main pour une 2-3 variables, vérifier
par différences finies.""", ),
code("""def f(xy):
    x, y = xy
    return np.array([x**2 * y, np.sin(x * y), x + y**3])

def J_f(x, y):
    return np.array([[2*x*y,        x**2],
                     [y*np.cos(x*y), x*np.cos(x*y)],
                     [1,            3*y**2]])

pt = np.array([0.7, -0.4])
J = J_f(*pt)
h = 1e-6
J_num = np.zeros((3, 2))
for j in range(2):
    dp = np.zeros(2); dm = np.zeros(2); dp[j] = h; dm[j] = -h
    J_num[:, j] = (f(pt + dp) - f(pt + dm)) / (2 * h)
print("J (analytique) =\\n", J)
print("J (numérique)  =\\n", J_num)
assert np.allclose(J, J_num, atol=1e-5)

# Chaîne : f(x) = ||g(x)||², g(x,y) = (sin(x), x·y)
# d(f∘g)/dx = 2 g· g'  ->  J_{f∘g} = J_f(g) @ J_g
x, y = 0.6, 0.3
def gfun(xy):
    x, y = xy
    return np.array([np.sin(x), x * y])
Jg = np.array([[np.cos(x), 0.0], [y, x]])   # Jacobien de g
gval = gfun((x, y))
# h(x,y) = ||g(x,y)||²  ;  dh/dx = 2 * g · (dg/dx)
dhdx = 2.0 * (gval[0] * Jg[0, 0] + gval[1] * Jg[1, 0])
hfun = lambda xy: np.linalg.norm(gfun(xy)) ** 2
dhdx_num = (hfun((x + h, y)) - hfun((x - h, y))) / (2 * h)
print(f"\\nChaîne : dh/dx (analytique) = {dhdx:.5f}")
print(f"          (numérique)        = {dhdx_num:.5f}")
assert np.isclose(dhdx, dhdx_num, atol=1e-5)"""),
md("""## 8 · Backpropagation & dérivées automatiques *(MML ch. 5.6)*

La **rétropropagation** n'est qu'une application *efficace* de la règle
de la chaîne sur un graphe de calcul : on propage les dérivées de la
perte vers les paramètres, en réutilisant les valeurs du *forward pass*.

Complexité : **O(définition du réseau)** — contre O(poids) si on
recalcule chaque dérivée par différences finies.

**Gradient check** (bonne pratique d'implémentation) : comparer le
gradient analytique (backprop) au gradient numérique (différence
centrale). S'ils ne sont pas égaux à `1e-6` près → bug dans la
backprop.""", ),
code("""rng = np.random.default_rng(0)
x_in = np.array([0.5, -0.3])
y_true = 1.0
W1 = np.array([[0.5, -0.2], [0.1, 0.8], [-0.4, 0.6]])  # 2 -> 3
b1 = np.array([0.1, -0.2, 0.0])
W2 = np.array([0.3, -0.7, 0.9])                        # 3 -> 1
b2 = 0.2

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def net_loss(W1, b1, W2, b2):
    z1 = x_in @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return (a2 - y_true) ** 2, a1, a2

loss, a1, a2 = net_loss(W1, b1, W2, b2)
# ---- backprop (chaîne de la règle) ----
delta_out = 2 * (a2 - y_true) * a2 * (1 - a2)      # dL/da2
dW2 = delta_out * a1                               # da2/dW2 = a1
db2 = delta_out
dz1 = (W2 * delta_out) * (1 - a1 ** 2)             # da1/dz1 = 1 - tanh²
dW1 = np.outer(x_in, dz1)
db1 = dz1
# ---- gradient check numérique sur W1 ----
h = 1e-6
dW1_num = np.zeros_like(W1)
for i in range(2):
    for j in range(3):
        dp = W1.copy(); dm = W1.copy()
        dp[i, j] += h; dm[i, j] -= h
        dW1_num[i, j] = (net_loss(dp, b1, W2, b2)[0]
                         - net_loss(dm, b1, W2, b2)[0]) / (2 * h)
print("dL/dW1 analytique :\\n", dW1)
print("dL/dW1 numérique  :\\n", dW1_num)
assert np.allclose(dW1, dW1_num, atol=1e-6)
print("\\nGradient check OK : backprop = règle de la chaîne.")"""),
md("""## 9 · Hessian & linéarisation de Taylor *(MML ch. 5.7-5.8)*

**Linéarisation de Taylor d'ordre 2** autour de `x₀` :
```
f(x₀ + v) ≈ f(x₀) + ∇f(x₀)ᵀ v + ½ vᵀ H(x₀) v
```
Outils :
- **Hessian** `H` : matrice des 2ndes dérivées ; signature des valeurs
  propres → minimum local (SP), maximum (SN), point selle (indéfinie)
- **pas de Newton** : `x ← x - H⁻¹ ∇f` (convergence quadratique près
  du minimum, mais coûteux)
- **Hessian-vector (HVP)** : `H @ v` sans construire H — utilisé par
  L-BFGS, natural gradient, et la double backpropagation""", ),
code("""# f(x) = ½ xᵀAx - bᵀx (quadratique) : Taylor ordre 2 EXACT
Aq = np.array([[3., 1.], [1., 2.]]); bq = np.array([1., -1.])
fq = lambda v: 0.5 * v @ Aq @ v - bq @ v
x0 = np.array([0.3, -0.2]); v = np.array([0.5, 0.4])
grad = Aq @ x0 - bq
H = Aq
f_true = fq(x0 + v)
T1 = fq(x0) + grad @ v
T2 = fq(x0) + grad @ v + 0.5 * v @ H @ v
print(f"f(x0+v)        = {f_true:.5f}")
print(f"Taylor ordre 1 = {T1:.5f}   (erreur {abs(f_true-T1):.4f})")
print(f"Taylor ordre 2 = {T2:.5f}   (erreur {abs(f_true-T2):.2e} -> exact pour quadratique)")
assert np.isclose(T2, f_true)

# Pas de Newton : convergence en 1 pas pour une quadratique
x_new = x0 - np.linalg.solve(H, grad)
print(f"\\nPas Newton : x* = {x_new} (minimum exact de f quadratique)")
assert np.allclose(x_new, np.linalg.solve(Aq, bq))

# Hessian d'une fonction NON quadratique : test de signature
# f(x,y) = x⁴/4 - x² + xy + y⁴/4  ->  H = [[3x²-2, 1],[1, 3y²]]
def H_analytique(x, y):
    return np.array([[3*x**2 - 2, 1.0], [1.0, 3*y**2]])
Hpt = H_analytique(1.0, -0.5)
print("\\nH en (1, -0.5) =\\n", Hpt, " valeurs propres =", np.linalg.eigvalsh(Hpt))
print("H indéfini (une +, une -) -> POINT SELLE")
assert np.linalg.eigvalsh(Hpt)[0] < 0 < np.linalg.eigvalsh(Hpt)[1]

# HVP par différences du gradient (double backprop en pratique)
def f4(x, y):
    return x**4/4 - x**2 + x*y + y**4/4
def g4(x, y):
    return np.array([x**3 - 2*x + y, x + 3*y**3])
hp = 1e-4
vp = np.array([0.7, -0.3])
Hvp = H_analytique(1.0, -0.5) @ vp
Hvp_num2 = (g4(1.0 + hp*vp[0], -0.5 + hp*vp[1]) - g4(1.0 - hp*vp[0], -0.5 - hp*vp[1])) / (2*hp)
print(f"H @ v (analytique) = {Hvp},  HVP numérique = {Hvp_num2}")
assert np.allclose(Hvp, Hvp_num2, atol=1e-4)"""),
md("""## 10 · Optimisation contrainte & multiplicateurs de Lagrange *(MML ch. 7.2)*

Problème : minimiser `f(x)` sous contrainte `g(x) = 0`.
Idea : au point optimal, **le gradient de f est colinéaire au gradient
de g** (sinon on pourrait descendre f en se déplaçant le long de la
variété de contrainte).

**Théorème de Lagrange** : il existe `λ` tel que
```
∇f(x*) + λ ∇g(x*) = 0      (conditions d'optimalité, KKT dans le cas général)
```
On résout le système d'équations `∇f + λ∇g = 0` + `g(x*) = 0`
(`n+1` équations, `n+1` inconnues).

Pour les **inégalités** `g(x) ≤ 0`, on ajoute les conditions de **complémentarité** :
`λ ≥ 0`, `λ g(x*) = 0` (la contrainte est soit inactive λ=0, soit active
`g(x*)=0`). C'est la base des SVM (T07 / MML ch. 12) et de la régularisation.""", ),
code("""# Exemple 1 : min x² + y²  s.t.  x + y = 2
# ∇f = λ∇g : (2x, 2y) = λ(1, 1) -> x = y ;  x + y = 2 -> x* = y* = 1, f* = 2, λ* = 2
from scipy.optimize import minimize
res = minimize(lambda vv: vv[0]**2 + vv[1]**2, x0=[5., 0.],
               constraints={'type': 'eq', 'fun': lambda vv: vv[0] + vv[1] - 2},
               method='SLSQP')
print("Lagrange : x*=1, y*=1, f*=2 (λ*=2)")
print("scipy SLSQP :", res.x, " valeur :", res.fun)
assert np.allclose(res.x, [1, 1]) and np.isclose(res.fun, 2)

# Exemple 2 (inégalité, complémentarité) :
# min (x-3)² + (y-4)²  s.t.  x² + y² ≤ 1
# minimum sans contrainte (3,4) est hors du disque -> optimum SUR le bord
# point le plus proche de (3,4) sur le cercle : (3,4)/||(3,4)|| = (1.8, 2.4)
# f* = ||(3,4)||² - 1 = 25 - 1 = 24 ? -> attention : f = 25 - 2*(3*1.8+4*2.4)/5 + 1
res2 = minimize(lambda vv: (vv[0]-3)**2 + (vv[1]-4)**2, x0=[0.9, 0.1],
                constraints=[{'type': 'ineq', 'fun': lambda vv: 1 - vv[0]**2 - vv[1]**2}],
                method='SLSQP')
print("\\nInégalité :", res2.x, " valeur :", res2.fun)
assert np.allclose(res2.x, [1.8, 2.4]) and np.isclose(res2.fun, 16.0)
print("f* = 16 : le gradient de f en (1.8,2.4) est colinaire au gradient de g")
gf = 2 * np.array([1.8 - 3, 2.4 - 4])
gg = 2 * np.array([1.8, 2.4])
assert np.allclose(gf, (gf @ gg / (gg @ gg)) * gg, atol=1e-6)  # colinéarité"""),
]

# ======================================================================
# 04 — PROBABILITÉS
# ======================================================================
new04 = [
md("""## 8 · Familles exponentielles & conjugaison *(MML ch. 6.6)*

Une loi est dans la **famille exponentielle** si sa densité s'écrit
```
p(x | θ) = h(x) · exp( η(θ)ᵀ T(x) - A(θ) )
```
avec `η` **paramètre naturel**, `T` **statistique suffisante**, `A`
fonction de normalisation. Les lois de base (Bernoulli, Gaussienne,
Poisson, Gamma...) sont toutes dans cette famille.

**Propriété clé — conjugaison** : si le prior `p(θ)` et la vraisemblance
`p(x|θ)` sont dans la même famille exponentielle (conjugués), alors le
postérieur est de la **même famille**, avec mise à jour analytique des
paramètres. C'est ce qui rend le bayésien **calculable** en pratique.

| Vraisemblance | Prior conjugué | Posterior |
|---|---|---|
| Bernoulli | Bêta | Bêta |
| Binomiale | Bêta | Bêta |
| Gaussienne (σ connu) | Gaussienne | Gaussienne |
| Poisson | Gamma | Gamma |
| Exponentielle | Gamma | Gamma |
| Gamma | Gamma | Gamma |""", ),
code("""# Conjugaison Poisson-Gamma : λ ~ Gamma(α, β), x ~ Poisson(λ)
# -> λ | x ~ Gamma(α + Σx, β + n)
from scipy import stats
rng = np.random.default_rng(1)
alpha, beta = 2.0, 1.0     # prior Gamma(shape=2, rate=1), E[λ]=2
lam_true = 5.0
x = rng.poisson(lam_true, 6)
print("Observations Poisson(5):", x)
print(f"E[λ prior] = {alpha/beta:.3f}")
print(f"MLE de λ   = {x.mean():.3f}")
print(f"E[λ post]  = {(alpha + x.sum()) / (beta + len(x)):.3f}")

# Le postérieur s'approche du MLE quand n grandit
for n_obs in [6, 60, 600]:
    xo = rng.poisson(lam_true, n_obs)
    e_post = (alpha + xo.sum()) / (beta + n_obs)
    print(f"  n={n_obs:4d} : E[λ post] = {e_post:.3f}  (MLE = {xo.mean():.3f})")

# Vérif par simulation du postérieur
post = stats.gamma(alpha + x.sum(), scale=1.0 / (beta + len(x)))
print(f"\\nPosterior Gamma({alpha + x.sum():.0f}, rate={beta + len(x):.0f}) : "
      f"E={post.mean():.3f}, sd={post.std():.3f}")
assert np.isclose(post.mean(), (alpha + x.sum()) / (beta + len(x)))"""),
md("""## 9 · Changement de variable & échantillonnage inverse *(MML ch. 6.7)*

Si `Y = g(X)` est une bijection, la densité de `Y` :
```
f_Y(y) = f_X(g⁻¹(y)) · |d/dy g⁻¹(y)|
```
Le **jacobien** (dét. de la matrice des dérivées) est l'analogue
multivarié du terme `|g⁻¹'(y)|`.

Applications :
- **échantillonnage par transformation inverse** : si `X ~ U(0,1)`
  alors `Y = F_Y⁻¹(X)` suit la loi de densité `F_Y`
- **normalisation de volume** en transformations (NFT, flow)
- **dérivation de lois** : `Y = -ln(U)` avec `U ~ U(0,1)` donne
  `Y ~ Exponentielle(1)""", ),
code("""# Y = -ln(X) pour X ~ U(0,1)  ->  Y ~ Exponentielle(1)
rng = np.random.default_rng(7)
X = rng.uniform(0, 1, 200000)
Y = -np.log(X)
ys = np.linspace(0, 5, 50)
hist, edges = np.histogram(Y, bins=ys, density=True)
mid = (edges[:-1] + edges[1:]) / 2
err = np.max(np.abs(hist - np.exp(-mid)))
print(f"Max |densité empirique - exp(-y)| = {err:.4f}  (Y ~ Exponentielle(1))")
assert err < 0.03

# Échantillonnage inverse pour N(μ, σ²) : y = μ + σ Φ⁻¹(u), u ~ U(0,1)
Z = stats.norm.ppf(rng.uniform(0, 1, 100000))
print(f"N(0,1) par transformation inverse : mean={Z.mean():.3f}, sd={Z.std():.3f}")
assert abs(Z.mean()) < 0.05 and abs(Z.std() - 1) < 0.05

# Multivarié : jacobien pour la transformation polaire
# (x,y) -> (r,θ) : |J| = r  ->  si (x,y) ~ N(0,I) alors r ~ Rayleigh
xy = rng.normal(0, 1, (100000, 2))
r = np.linalg.norm(xy, axis=1)
# densité de r : f(r) = r·exp(-r²/2)
rs = np.linspace(0, 4, 40)
hist_r, edges_r = np.histogram(r, bins=rs, density=True)
mid_r = (edges_r[:-1] + edges_r[1:]) / 2
err_r = np.max(np.abs(hist_r - mid_r * np.exp(-mid_r**2 / 2)))
print(f"Rayleigh : max err densité = {err_r:.4f}")
assert err_r < 0.03"""),
]

# ======================================================================
# 05 — MODÉLISATION STATISTIQUE
# ======================================================================
new05 = [
md("""## 7 · Régression linéaire bayésienne *(MML ch. 9.3)*

Le T05 §5 a donné la régression **frequentiste** (maximum de vraisemblance).
La version **bayésienne** traite les poids `w` comme aléatoires :
- **prior** : `w ~ N(0, α⁻¹ I)` (contraints autour de 0 → effet de Ridge)
- **bruit** : `y = Xw + ε`, `ε ~ N(0, β⁻¹ I)`
- **posterior** : `w | D ~ N(m_N, S_N)` avec
```
S_N⁻¹ = α I + β XᵀX
m_N   = S_N (α m₀ + β Xᵀy)
```
- **prédiction** : `y* | x*, D ~ N(m_Nᵀx*, 1/β + x*ᵀ S_N x*)`

**Pourquoi c'est utile :**
1. le prior `α I` = **régularisation Ridge** (liage direct avec T07)
2. le **doubt quantifié** : la variance prédictive croît avec l'écart
   au nuage de données
3. cadre unifié : le MLE correspond au cas de prior uniforme (α→0)""", ),
code("""rng = np.random.default_rng(3)
n = 100
X = rng.normal(0, 1, (n, 1))
beta_true = np.array([1.5])
y = X @ beta_true + 0.0 + rng.normal(0, 0.5, n)

ols = lm.LinearRegression().fit(X, y)
print(f"OLS :  β̂ = {ols.coef_[0]:.3f} (vrai 1.5)")

def bayes_linear_reg(X, y, alpha, beta, m0=0.0):
    S0_inv = alpha * np.eye(X.shape[1])
    S_N_inv = S0_inv + beta * (X.T @ X)
    S_N = np.linalg.inv(S_N_inv)
    m_N = S_N @ (S0_inv @ m0 + beta * (X.T @ y))
    return m_N, S_N

beta_noise = 4.0   # 1/σ² avec σ=0.5
for a, lbl in [(0.1, "prior faible (α=0.1)"), (1000.0, "prior fort (α=1000, sd=0.03)")]:
    m_N, S_N = bayes_linear_reg(X, y, a, beta_noise)
    print(f"{lbl} : β̂_bayes = {m_N[0]:.3f} ± {np.sqrt(S_N[0,0]):.3f}")

# Vérif : prior faible ~ OLS ; prior fort tire vers 0
m_weak, _ = bayes_linear_reg(X, y, 0.1, beta_noise)
m_strong, S_strong = bayes_linear_reg(X, y, 1000.0, beta_noise)
assert abs(m_weak[0] - ols.coef_[0]) < 0.05
assert abs(m_strong[0]) < 0.6   # tiré fortement vers 0 (Ridge extrême)

# Incertitude prédictive à x* = 0 (dans le nuage) vs x* = 3 (loin)
x_near = np.array([[0.0]])
x_far = np.array([[3.0]])
for xx, lbl in [(x_near, "x*=0 (dans le nuage)"), (x_far, "x*=3 (loin)")]:
    pred_var = 1 / beta_noise + float(xx @ S_N @ xx.T)
    print(f"var prédictive {lbl} : {pred_var:.4f}")
print("Plus on est loin des données, plus la prédiction est incertaine.")"""),
]

# ======================================================================
# APPLIQUE LES INSERTIONS
# ======================================================================
def insert_before(nb, anchor_prefix, new_cells):
    idx = None
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and c.source.strip().startswith(anchor_prefix):
            idx = i
            break
    assert idx is not None, f"anchor '{anchor_prefix}' introuvable"
    nb.cells[idx:idx] = new_cells
    return idx

def rename_source(nb, old, new):
    for c in nb.cells:
        if c.source.startswith(old):
            c.source = c.source.replace(old, new, 1)
            break

def load(name):
    return nbf.read(os.path.join(NB_DIR, name), as_version=4)

def save(nb, name):
    nbf.write(nb, os.path.join(NB_DIR, name))

# 01 : insérer avant "## 🎯 Exercices corrigés"
nb01 = load("01_algebre_lineaire.ipynb")
insert_before(nb01, "## 🎯 Exercices corrigés", new01)
save(nb01, "01_algebre_lineaire.ipynb")
print("01 : +3 sections (eigendecomp, Cholesky, PCA latente)")

# 03 : insérer avant "## 7 · Récapitulatif" et renumber en 11
nb03 = load("03_calcul_optimisation.ipynb")
insert_before(nb03, "## 7 · Récapitulatif", new03)
rename_source(nb03, "## 7 · Récapitulatif", "## 11 · Récapitulatif")
save(nb03, "03_calcul_optimisation.ipynb")
print("03 : +4 sections (Jacobian, backprop, Hessian/Taylor, Lagrange); renumber recap -> 11")

# 04 : insérer avant "## 🎯 Exercices corrigés"
nb04 = load("04_probabilités.ipynb")
insert_before(nb04, "## 🎯 Exercices corrigés", new04)
save(nb04, "04_probabilités.ipynb")
print("04 : +2 sections (famille exponentielle, changement de variable)")

# 05 : insérer avant "## 🎯 Exercices corrigés"
nb05 = load("05_modelisation_statistique.ipynb")
insert_before(nb05, "## 🎯 Exercices corrigés", new05)
save(nb05, "05_modelisation_statistique.ipynb")
print("05 : +1 section (régression bayésienne)")
