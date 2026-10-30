#!/usr/bin/env python3
# Build notebooks 07, 08, 09 for ADIF72
import nbformat as nbf
import os

def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
def make_nb(cells, title):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": title, "provenance": []},
        "accelerator": "GPU"
    }
    nb.cells = cells
    return nb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======================================================================
# NOTEBOOK 07 — APPLICATIONS EN SCIENCE DES DONNÉES
# ======================================================================
cells07 = [
md("""# T07 · Applications en science des données — ADIF72

**Objectifs :**
- Pipeline de modélisation prédictive complet
- Validation des modèles (k-fold, courbes d'apprentissage)
- Biais-variance, sur/sous-apprentissage
- Métriques d'évaluation (RMSE, R², précision, rappel, F1, AUC)
- Étude de cas end-to-end + biais & éthique

**Lien avec le reste :** ce topic assemble T01-T06 en pratique.
Toutes les notions (SVD→PCA, gradient→entraînement, tests→validation)
trouvent ici leur application.""", ),
code("""import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     KFold, learning_curve)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                            precision_score, recall_score, f1_score,
                            roc_auc_score)
np.set_printoptions(precision=4, suppress=True, linewidth=100)
print("Prêt. sklearn importé.")
rng = np.random.default_rng(5)"""),
md("""## 1 · Pipeline de modélisation régressive (end-to-end)

Les **6 étapes** d'un pipeline classique :
1. **Données** : chargement, exploration
2. **Séparation** train/test (jamais mélanger !)
3. **Prétraitement** : échelle, encodage (fit SUR LE TRAIN SEUL)
4. **Entraînement** du modèle
5. **Évaluation** sur le test (une seule fois)
6. **Déploiement / itération**

**Piège n°1 : data leakage** — le prétraitement (ex. `StandardScaler`)
doit être `fit` sur les données train et `transform` sur le test.
Sinon le modèle « voit » la moyenne du test → évaluation trop optimiste.
En production, on encapsule prétraitement+modèle dans un `Pipeline`.""", ),
code("""n = 500
superficie = rng.uniform(50, 200, n)
age = rng.integers(0, 50, n)
nb_pieces = rng.integers(2, 7, n)
prix = (20000 * superficie - 1500 * age + 8000 * nb_pieces
        + rng.normal(0, 30000, n))
X = pd.DataFrame({'superficie': superficie, 'age': age, 'nb_pieces': nb_pieces})
y = pd.Series(prix)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=5)
scaler = StandardScaler().fit(X_tr)          # fit SUR LE TRAIN
X_tr_s = scaler.transform(X_tr); X_te_s = scaler.transform(X_te)

model = LinearRegression().fit(X_tr_s, y_tr)
pred = model.predict(X_te_s)
rmse = np.sqrt(mean_squared_error(y_te, pred))
r2 = r2_score(y_te, pred)
print(f"RMSE = {rmse:,.0f}  (bruit ~30k)")
print(f"R²   = {r2:.4f}")
assert r2 > 0.99 and rmse < 40000"""),
md("""## 2 · Validation k-fold & biais-variance

Le **train/test split** dépend de la chance de la coupe. Le **k-fold
cross-validation** est plus robuste : on coupe les données en `k` parts,
on entraîne `k` fois (à chaque fois sur `k-1` parts, test sur la restante),
puis on moyenne les scores.

**Courbe d'apprentissage** (learning curve) : on varie la taille du train
et on regarde l'écart train/validation.
- Écart grand → **sur-apprentissage** (variance)
- Deux courbes basses → **sous-apprentissage** (biais)
- Courbes qui convergent en haut → bon modèle""", ),
code("""# Données non linéaires (sin)
Xnl = rng.uniform(0, 1, (300, 1))
ynl = np.sin(6 * np.pi * Xnl[:, 0]) + rng.normal(0, 0.1, 300)

lin = LinearRegression()
forest = RandomForestRegressor(n_estimators=50, random_state=5)
for name, m in [("Linéaire", lin), ("Forêt", forest)]:
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    scores = cross_val_score(m, Xnl, ynl, cv=kf, scoring='r2')
    print(f"{name:8s} : R² moyen = {scores.mean():.3f} (±{scores.std():.3f})")

print("\\nLe modèle trop simple (linéaire) sous-apprend sur du sin ;")
print("la forêt capture la non-linéarité (compromis biais/variance).")"""),
md("""## 3 · Sur-apprentissage explicite (biais-variance)

Le **trade-off** : plus le modèle est flexible, moins il a de biais
mais plus il a de variance.
- **Biais** : erreur due aux hypothèses trop fortes du modèle (sous-apprend)
- **Variance** : sensibilité aux fluctuations de l'échantillon (sur-apprend)
- **MSE = biais² + variance + bruit**

On peut le visualiser avec un polynôme dont on augmente le degré :
au début le modèle est trop rigide (sous-apprend), puis il colle,
puis il épouse le bruit (sur-apprend).""", ),
code("""x = rng.uniform(-3, 3, 100)
y = np.cos(x) + rng.normal(0, 0.1, 100)
Xpoly = x.reshape(-1, 1)
Xtr, Xte, ytr, yte = train_test_split(Xpoly, y, test_size=0.3, random_state=5)

print("Degré -> R² train vs R² test :")
for deg in [1, 3, 8, 20]:
    poly = PolynomialFeatures(deg)
    Ptr = poly.fit_transform(Xtr); Pte = poly.transform(Xte)
    m = LinearRegression().fit(Ptr, ytr)
    r2_tr = m.score(Ptr, ytr); r2_te = m.score(Pte, yte)
    print(f"  degré {deg:2d} : train={r2_tr:.3f}  test={r2_te:.3f}"
          f"  {'<-- sur-apprend' if r2_tr - r2_te > 0.3 else ''}")

print("\\nBas degré : sous-apprend (biais). Haut degré : sur-apprend (variance).")"""),
md("""## 4 · Classification : métriques

L'**accuracy** est trompeuse en présence de **déséquilibre** :
si 90% des cas sont négatifs, prédire toujours « négatif » donne
accuracy = 0.9 mais F1 ≈ 0. On utilise alors :
- **Précision** : P(déclaré positif | positif) — qualité des alertes
- **Rappel** : P(déclaré positif | vraiment positif) — sensibilité
- **F1** : moyenne harmonique précision/rappel
- **AUC-ROC** : qualité du classement (seuil-invariant) — la plus robuste

La **courbe ROC** trace rappel vs (1-précision) pour tous les seuils ;
AUC = aire sous la courbe.""", ),
code("""# Données équilibrées
Xc = rng.normal(0, 1, (400, 2))
Xc[:200] += np.array([2, 0]); Xc[200:] -= np.array([2, 0])
yc = np.array([1]*200 + [0]*200)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3, random_state=5)
sc = StandardScaler().fit(Xc_tr)
clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xc_tr), yc_tr)
pred = clf.predict(sc.transform(Xc_te))
prob = clf.predict_proba(sc.transform(Xc_te))[:, 1]
acc = accuracy_score(yc_te, pred); prec = precision_score(yc_te, pred)
rec = recall_score(yc_te, pred); f1 = f1_score(yc_te, pred)
auc = roc_auc_score(yc_te, prob)
print(f"Accuracy={acc:.3f}  Précision={prec:.3f}  Rappel={rec:.3f}  "
      f"F1={f1:.3f}  AUC={auc:.3f}")
assert acc > 0.95 and auc > 0.99

# Données déséquilibrées (90/10)
Xd = rng.normal(0, 1, (1000, 2)); yd = np.zeros(1000, int)
Xd[:100] += np.array([3, 0]); yd[:100] = 1
Xdt, Xde, ydt, yde = train_test_split(Xd, yd, test_size=0.3, random_state=5)
scd = StandardScaler().fit(Xdt)
clf2 = LogisticRegression(max_iter=1000).fit(scd.transform(Xdt), ydt)
pred2 = clf2.predict(scd.transform(Xde))
print(f"\\nDéséquilibrées : Accuracy={accuracy_score(yde,pred2):.3f} "
      f"(trompeuse !)  F1={f1_score(yde,pred2,zero_division=0):.3f}")
print("-> préférer F1 / AUC à l'accuracy.")"""),
md("""## 5 · Biais & considérations éthiques

Un modèle n'est que **aussi bon que ses données**. Les biais classiques :
- **Biais de sélection** : l'échantillon ne représente pas la population
  (ex : seulement les clients actifs). Le modèle est mauvais hors-échantillon.
- **Biais d'historique** : les données encodent les discriminations passées ;
  le modèle les reproduit.
- **Biais de mesure** : la variable cible est mal définie.

**Vérifications systématiques** avant de déployer :
- Couverture des variables de prédiction dans le jeu de test
- Équité (parité démographique, égalité des faux positifs par groupe)
- Sensibilité aux valeurs extrêmes""", ),
code("""# Biais de sélection : entraînement sur x>0 seulement
Xfull = rng.uniform(-5, 5, 2000)
yfull = np.sin(Xfull) + rng.normal(0, 0.1, 2000)
mask = Xfull > 0
Xb, yb = Xfull[mask].reshape(-1, 1), yfull[mask]
mb = LinearRegression().fit(Xb, yb)
err_biais = np.mean((mb.predict(Xfull[~mask].reshape(-1,1)) - yfull[~mask])**2)
print(f"Modèle biaisé (entraîné sur x>0) : coef={mb.coef_[0]:.3f}")
print(f"RMSE sur la partie non vue (x<0) = {np.sqrt(err_biais):.3f}")
print("Leçon : échantillon non représentatif = biais systématique.")"""),
md("""## 6 · Tableau récapitulatif — choisir la bonne méthode
"""),
md("""| Situation | Méthode |
|---|---|
| n grande, x ~ normale | tests paramétriques (z, t) |
| n petit / non normal | tests non paramétriques |
| 3+ groupes (moyennes) | ANOVA + Tukey |
| indépendance (catégories) | chi-deux |
| prédiction continue | régression (lin, arbres, forêt) |
| prédiction binaire | régression logistique / arbres |
| données déséquilibrées | F1, AUC (pas l'accuracy) |
| sur-apprentissage | régularisation, plus de données |
| biais d'échantillonnage | stratification, vérif couverture |
"""),
code("""print("✅ T07 terminé.")"""),
]

# ======================================================================
# NOTEBOOK 08 — PRÉPARATION EXAMEN
# ======================================================================
cells08 = [
md("""# T08 · Préparation examen (Devoir écrit 2 h, papier)

**Rappel du barème ADIF72 :**
- **Devoir écrit (DE)** : individuel, 2 h, **sur papier**, **60 %**
- **Interrogation de TD** : groupe, **40 %**

Ce notebook regroupe des **drills à la main** : énoncé, méthode, corrigé
pas-à-pas. **À recopier à la main sans Python** pour s'entraîner au DE.
Les cellules de vérification (Python) servent à contrôler vos résultats.

**Structure simulée du DE (120 min) :**
- Partie A (30 min) : algèbre linéaire + géométrie
- Partie B (30 min) : calcul/optimisation + probabilités
- Partie C (60 min) : statistique + tests + régression""", ),
code("""import numpy as np
from scipy import stats
np.set_printoptions(precision=4, suppress=True, linewidth=100)
print("Prêt pour le DE.")"""),
md("""## PARTIE A — Algèbre & géométrie (30 min)""", ),
code("""# A1 : système par Gauss
#   2x + y - z = 4 ;  x + 2y + z = 5 ;  x - y + 2z = 1
# Corrigé : x=1.5, y=1.5, z=0.5
A = np.array([[2,1,-1],[1,2,1],[1,-1,2]]); b = np.array([4,5,1])
sol = np.linalg.solve(A, b)
print("A1 : solution =", sol, "(= 1.5, 1.5, 0.5)")
assert np.allclose(sol, [1.5, 1.5, 0.5])"""),
code("""# A2 : det, inverse
M = np.array([[4,2],[3,1]])
print("A2 : det =", np.linalg.det(M), " M^-1 =\\n", np.linalg.inv(M))
assert np.isclose(np.linalg.det(M), -2)
assert np.allclose(M @ np.linalg.inv(M), np.eye(2))"""),
code("""# A3 : valeurs propres de [[3,1],[1,3]]
P = np.array([[3,1],[1,3]])
e, _ = np.linalg.eigh(P)
print("A3 : valeurs propres =", e, "(= 4 et 2)")
assert np.allclose(sorted(e), [2, 4])"""),
code("""# A4 : angle entre u=(2,1,2) et v=(1,-1,1)
u5 = np.array([2.,1,2]); v5 = np.array([1.,-1,1])
ang = np.degrees(np.arccos(u5@v5/(np.linalg.norm(u5)*np.linalg.norm(v5))))
print(f"A4 : angle = {ang:.2f}° (≈ 54.74°)")
assert np.isclose(ang, np.degrees(np.arccos(1/np.sqrt(3))))"""),
md("""## PARTIE B — Calcul, optimisation & probabilités (30 min)""", ),
code("""# B1 : gradient + point critique de g=x²+y²-4x+2y-1
# ∇g = (2x-4, 2y+2)=0 -> x=2, y=-1 ; Hessian = 2I >0 -> MINIMUM ; g(2,-1)=-6
g = lambda x,y: x*x+y*y-4*x+2*y-1
print(f"B1 : gradient(2,-1)=({2*2-4},{2*(-1)+2}) ; g(2,-1)={g(2,-1)}")
assert g(2,-1) == -6"""),
code("""# B2 : descente de gradient f(x)=(x-3)², x0=0, α=0.5
x0, alpha = 0.0, 0.5
x1 = x0 - alpha*2*(x0-3)   # = 3 (déjà au minimum)
print(f"B2 : x1 = {x1}, f(x1) = {(x1-3)**2} (minimum atteint)")
assert x1 == 3.0"""),
code("""# B3 : Bayes  P(A)=0.4, P(B|A)=0.7, P(B|Aᶜ)=0.3
pB = 0.7*0.4 + 0.3*0.6
pA_B = 0.7*0.4/pB
print(f"B3 : P(B)={pB:.4f}, P(A|B)={pA_B:.4f} (≈ 0.6087)")
assert np.isclose(pA_B, 14/23)"""),
code("""# B4 : normale X~N(60,10)  ->  P(50<X<70), P(X>80), P95
print(f"B4 : P(50<X<70)={stats.norm.cdf(1)-stats.norm.cdf(-1):.4f} (≈0.6827)")
print(f"      P(X>80)={1-stats.norm.cdf(2):.4f} (≈0.0228)")
print(f"      P95={60+10*stats.norm.ppf(0.95):.3f} (≈76.45)")
assert np.isclose(stats.norm.cdf(1)-stats.norm.cdf(-1), 0.6826, atol=1e-3)"""),
code("""# B5 : binomiale n=10, p=0.5  ->  P(X=5), P(X≥8)
print(f"B5 : P(X=5)={stats.binom.pmf(5,10,0.5):.4f} (≈0.2461)")
print(f"      P(X≥8)={1-stats.binom.cdf(7,10,0.5):.4f} (≈0.0547)")
assert np.isclose(stats.binom.pmf(5,10,0.5), 252/1024)"""),
md("""## PARTIE C — Statistique, tests, régression (60 min)""", ),
code("""# C1 : moyenne, variance, sd de [2,4,4,4,5,5,7,9]
xc = np.array([2,4,4,4,5,5,7,9], dtype=float)
print(f"C1 : moyenne={xc.mean()}, var={xc.var(ddof=1):.4f} (≈4.571), sd={xc.std(ddof=1):.4f}")
assert np.isclose(xc.mean(), 5) and np.isclose(xc.var(ddof=1), 32/7)"""),
code("""# C2 : IC95, n=25, X̄=10, s=2.5 (t de Student)
tcrit = stats.t.ppf(0.975, 24)
ic = (10 - tcrit*2.5/5, 10 + tcrit*2.5/5)
print(f"C2 : t={tcrit:.3f}, IC95=[{ic[0]:.3f},{ic[1]:.3f}] (≈ [8.97,11.03])")
assert np.isclose(ic[0], 8.968, atol=0.01)"""),
code("""# C3 : test t 2 échantillons  G1(n=30,52,5) vs G2(n=30,55,4)
se = np.sqrt(25/30 + 16/30); t = 3/se
print(f"C3 : t={t:.3f}, p={2*(1-stats.t.cdf(t,58)):.5f} -> rejet H0")
assert t > 2.5 and 2*(1-stats.t.cdf(t,58)) < 0.05"""),
code("""# C4 : régression (0,1),(1,2),(2,5),(3,4),(4,5) -> β, α, R²
xr = np.array([0,1,2,3,4],float); yr = np.array([1,2,5,4,5],float)
xb, yb = xr.mean(), yr.mean()
beta = np.sum((xr-xb)*(yr-yb))/np.sum((xr-xb)**2)
alpha = yb - beta*xb
r2 = 1 - np.sum((yr-(alpha+beta*xr))**2)/np.sum((yr-yb)**2)
print(f"C4 : β={beta:.3f}, α={alpha:.3f}, R²={r2:.4f}")
assert np.isclose(beta, 1.0) and np.isclose(alpha, 1.4)"""),
code("""# C5 : ANOVA 3 groupes
G1,G2,G3 = np.array([2.,3,2,3]), np.array([4.,5,4,4]), np.array([5.,6,5,4])
f_val, p_val = stats.f_oneway(G1, G2, G3)
print(f"C5 : F={f_val:.3f} (≈15.80), p={p_val:.4f} -> rejet (F>Fcrit=4.256)")
assert np.isclose(f_val, 15.80, atol=0.1) and p_val < 0.01"""),
code("""# C6 : chi-deux 2x2  (30,20 / 20,30)
obs = np.array([[30,20],[20,30]])
chi2, p, df, _ = stats.chi2_contingency(obs, correction=False)
print(f"C6 : chi2={chi2:.3f} (=4), p={p:.4f} -> rejet (4>3.841)")
assert np.isclose(chi2, 4.0) and p < 0.05"""),
md("""## Conseils examen (DE papier)
- Écrivez la **méthode** avant de calculer (le barème note la démarche).
- **Vérifiez toujours** : `A@x = b`, `det(M)·M⁻¹ = I`, `R² ∈ [0,1]`.
- Ne sautez pas d'étapes : chaque ligne de calcul compte.
- Si bloqué : poser les équations donne déjà 30-40 % du point.
- Le **TD (40 %)** se joue en groupe : savoir expliquer/justifier oralement.
"""),
code("""print("✅ T08 terminé. Tous les drills vérifiés.")"""),
]

# ======================================================================
# NOTEBOOK 09 — GRAPHIQUES
# ======================================================================
cells09 = [
md("""# T09 · Graphiques pédagogiques — ADIF72

Visualisations des concepts clés. Chaque figure illustre un point
théorique du cours. En Colab, les graphiques s'affichent **inline**
(`%matplotlib inline`).

Les 8 figures :
1. TCL — effet de la taille d'échantillon
2. Descente de gradient — trajectoire 2D
3. PCA — SVD + projection
4. Distributions — Normal / Binomiale / Exponentielle / Bêta
5. Bayes — prior Bêta → posteriors
6. Couverture des intervalles de confiance
7. Sur-apprentissage — biais-variance
8. Learning curves — linéaire vs forêt""", ),
code("""import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
rng = np.random.default_rng(7)
plt.rcParams['figure.figsize'] = (8, 5)
print("Matplotlib prêt.")"""),
md("""### Fig 1 — Théorème de la limite centrale""", ),
code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
mu, sigma = 10, 3
n_samps = 5000
for n, ax, title in [(1, axes[0], 'n=1 : distribution des tirages'),
                     (30, axes[1], f'n=30 : X̄ (~ N({mu},{sigma**2/30}))')]:
    data = rng.normal(mu, sigma, n_samps) if n==1 else rng.normal(mu, sigma, (n_samps, n)).mean(axis=1)
    ax.hist(data, bins=40, density=True, alpha=0.75, color='#2878b5')
    pdf = stats.norm.pdf(data, mu, sigma if n==1 else sigma/np.sqrt(n))
    ax.plot(data, pdf, 'r-', lw=2, label='Normale')
    ax.set_title(title, fontsize=11); ax.legend()
fig.suptitle('TCL : moyenne de n tirages', y=1.02)
plt.show()"""),
md("""### Fig 2 — Trajectoire de descente de gradient""", ),
code("""f2 = lambda x, y: (x-3)**2 + (y+1)**2 + 0.3*x*y
df2 = lambda x, y: np.array([2*(x-3)+0.3*y, 2*(y+1)+0.3*x])
xx, yy = np.meshgrid(np.linspace(-4,6,200), np.linspace(-6,4,200))
traj = np.array([[-4.0, 0.0]])
for _ in range(40):
    x, y = traj[-1]; g = df2(x, y)
    traj = np.vstack([traj, [x-0.5*g[0], y-0.5*g[1]]])
fig, ax = plt.subplots(figsize=(7,6))
cont = ax.contourf(xx, yy, f2(xx, yy), levels=20, cmap='viridis', alpha=0.8)
ax.plot(traj[:,0], traj[:,1], 'r-o', ms=4, lw=1.5, label='Trajectoire GD')
ax.plot(3, -1, 'r*', ms=15, label='Optimum (3,-1)')
fig.colorbar(cont, ax=ax, label='f(x,y)')
ax.set_title('GD sur f(x,y)=(x-3)²+(y+1)²+0.3xy'); ax.legend(loc='upper left')
plt.show()"""),
md("""### Fig 3 — PCA (SVD)""", ),
code("""pts = rng.normal(0,1,(300,2))*np.array([4,1.2])
Xc = pts - pts.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
pc1, pc2 = Vt[0], Vt[1]
proj1 = Xc @ pc1
var_expl = (S[0]**2)/np.sum(S**2)*100
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(pts[:,0], pts[:,1], s=8, alpha=0.4, color='#2878b5', label='Données')
ax.quiver(0,0, pc1[0]*8, pc1[1]*8, angles='xy', scale_units='xy', scale=1, color='red', lw=2, label=f'PC1 ({var_expl:.1f}%)')
ax.quiver(0,0, pc2[0]*4, pc2[1]*4, angles='xy', scale_units='xy', scale=1, color='orange', lw=2, label='PC2')
ax.scatter(proj1*pc1[0], proj1*pc1[1], s=3, alpha=0.15, color='red', label='Projection')
ax.set_title(f'PCA manuel (SVD) : PC1 = {var_expl:.1f}% de la variance'); ax.legend(); ax.set_aspect('equal')
plt.show()"""),
md("""### Fig 4 — Distributions usuelles""", ),
code("""fig, axes = plt.subplots(2, 2, figsize=(11, 8))
x = np.linspace(-4, 4, 300)
axes[0,0].plot(x, stats.norm.pdf(x), color='#2878b5'); axes[0,0].set_title('Normale N(0,1)')
axes[0,1].bar(np.arange(0,11), stats.binom.pmf(np.arange(0,11), 10, 0.4), color='#756bb1'); axes[0,1].set_title('Binomiale n=10,p=0.4')
axes[1,0].plot(x, stats.expon.pdf(x, scale=1), color='#c12526'); axes[1,0].set_xlim(0,6); axes[1,0].set_title('Exponentielle λ=1')
axes[1,1].plot(x, stats.beta.pdf(x, 2, 5), color='#31a354'); axes[1,1].set_xlim(0,1); axes[1,1].set_title('Bêta(2,5)')
fig.suptitle('Lois usuelles en science des données', y=1.02)
plt.show()"""),
md("""### Fig 5 — Mise à jour bayésienne (Bêta)""", ),
code("""fig, ax = plt.subplots(figsize=(7,5))
x = np.linspace(0,1,300)
ax.plot(x, stats.beta.pdf(x,1,1), lw=2, label='Prior Bêta(1,1)')
for s,t,col in [(5,100,'#2878b5'),(80,100,'#c12526'),(50,100,'#31a354')]:
    ax.plot(x, stats.beta.pdf(x,1+s,1+t-s), lw=2, label=f'Bêta({1+s},{1+t-s}) — {s}/{t}')
ax.set_xlim(0,1); ax.set_xlabel('p'); ax.set_title('Prior -> Posterior (Bayes)'); ax.legend(fontsize=9)
plt.show()"""),
md("""### Fig 6 — Couverture des intervalles de confiance""", ),
code("""mu_true, sigma_true = 0, 1
N = 2000
fig, ax = plt.subplots(figsize=(8,5))
for n in [10, 30, 100]:
    means, halves = [], []
    for _ in range(N):
        s = rng.normal(mu_true, sigma_true, n)
        mb, ss = s.mean(), s.std(ddof=1)
        means.append(mb); halves.append(stats.t.ppf(0.975, n-1)*ss/np.sqrt(n))
    means, halves = np.array(means), np.array(halves)
    cov = np.mean((means-halves<=0)&(means+halves>=0))
    ax.plot(means, means+halves, '|', color='#2878b5', alpha=0.15, ms=3)
    ax.plot(means, means-halves, '|', color='#2878b5', alpha=0.15, ms=3)
    print(f"n={n} : couverture = {cov:.4f} (≈ 95%)")
ax.axhline(0, color='r', lw=2, label='μ vrai = 0')
ax.set_title('Couverture des IC 95% (t de Student)'); ax.legend()
plt.show()"""),
md("""### Fig 7 — Biais-variance (degré polynôme)""", ),
code("""xp = rng.uniform(-3,3,200); yp = np.cos(xp)+rng.normal(0,0.1,200)
Xp = xp.reshape(-1,1); xx = np.linspace(-3,3,300).reshape(-1,1)
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(xp, yp, 'o', ms=4, alpha=0.5, color='#2878b5', label='Données')
ax.plot(xx, np.cos(xx), 'k-', lw=2, label='Vérité (cos)')
for deg in range(0, 16, 2):
    P = PolynomialFeatures(deg).fit_transform(Xp)
    m = LinearRegression().fit(P, yp)
    Pxx = PolynomialFeatures(deg).fit_transform(xx)
    ax.plot(xx, m.predict(Pxx), '-', lw=1, alpha=0.6, label=f'degré {deg}')
ax.set_title('Sur-apprentissage : polynôme riche = colle au bruit'); ax.legend(loc='lower right', fontsize=8, ncols=2)
plt.show()"""),
md("""### Fig 8 — Learning curves""", ),
code("""Xnl = rng.uniform(0,1,(400,1)); ynl = np.sin(6*np.pi*Xnl[:,0])+rng.normal(0,0.1,400)
fig, ax = plt.subplots(figsize=(8,5))
for name, m, col in [('Linéaire', LinearRegression(), '#31a354'),
                     ('Forêt', RandomForestRegressor(n_estimators=50, random_state=7), '#2878b5')]:
    lc = learning_curve(m, Xnl, ynl, cv=5, train_sizes=np.linspace(0.1,1.0,8), scoring='r2', random_state=7)
    ts, vs = lc[1].mean(axis=1), lc[2].mean(axis=1)
    tse, vse = lc[1].std(axis=1), lc[2].std(axis=1)
    ax.plot(lc[0], ts, 'o-', color=col, lw=2, label=f'{name} — train')
    ax.plot(lc[0], vs, 's-', color=col, lw=2, ls='--', label=f'{name} — validation')
    ax.fill_between(lc[0], ts-tse, ts+tse, alpha=0.15, color=col)
    ax.fill_between(lc[0], vs-vse, vs+vse, alpha=0.15, color=col)
ax.set_xlabel('Taille de l\\'entraînement'); ax.set_ylabel('R²')
ax.set_title('Learning curves : écart train/validation'); ax.legend(fontsize=9)
plt.show()"""),
code("""print("✅ T09 terminé. 8 figures générées.")"""),
]

specs = [
    ('07_applications_datascience.ipynb', cells07, 'ADIF72 - T07 Applications Datascience'),
    ('08_examen_preparation.ipynb', cells08, 'ADIF72 - T08 Preparation Examen'),
    ('09_graphiques.ipynb', cells09, 'ADIF72 - T09 Graphiques'),
]
for name, cells, title in specs:
    nb = make_nb(cells, title)
    path = os.path.join(BASE, 'notebooks', name)
    nbf.write(nb, path)
    print("wrote", path, "  cells:", len(cells))
