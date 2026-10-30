#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIF72 — T07 · APPLICATIONS EN SCIENCE DES DONNÉES
===================================================
- Modélisation prédictive (pipeline complet)
- Validation des modèles (k-fold, courbe d'apprentissage)
- Biais-variance, sure/sous-apprentissage
- Évaluation : métriques (RMSE, R², précision, F1, AUC)
- Étude de cas end-to-end (prédiction régressive)
- Biais et considérations éthiques en inférence

Lancer :  python3 code/07_applications_datascience.py
"""
import numpy as np
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


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


rng = np.random.default_rng(5)


# ----------------------------------------------------------------------
# 1) PIPELINE DE MODÉLISATION RÉGRESSIVE (end-to-end)
# ----------------------------------------------------------------------
sep("1. PIPELINE RÉGRESSION (prix de maison simulé)")
n = 500
superficie = rng.uniform(50, 200, n)                 # m²
age = rng.integers(0, 50, n)                          # années
nb_pieces = rng.integers(2, 7, n)
# prix = f des caractéristiques + bruit
prix = (20000 * superficie - 1500 * age + 8000 * nb_pieces
        + rng.normal(0, 30000, n))
X = pd.DataFrame({'superficie': superficie, 'age': age, 'nb_pieces': nb_pieces})
y = pd.Series(prix)

# Séparation train/test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=5)
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr); X_te_s = scaler.transform(X_te)

model = LinearRegression().fit(X_tr_s, y_tr)
pred = model.predict(X_te_s)
rmse = np.sqrt(mean_squared_error(y_te, pred))
r2 = r2_score(y_te, pred)
print(f"RMSE = {rmse:,.0f}  (bruit ~30k)")
print(f"R²   = {r2:.4f}  (≈1 : excellent ajustement)")
print(f"Coefs (données standardisées) = {dict(zip(X.columns, model.coef_.round(3)))}")
assert r2 > 0.99 and rmse < 40000


# ----------------------------------------------------------------------
# 2) VALIDATION k-FOLD + BIAIS-VARIANCE
# ----------------------------------------------------------------------
sep("2. VALIDATION K-FOLD")
# Jeu avec structure non linéaire (sin)
Xnl = rng.uniform(0, 1, (300, 1))
ynl = np.sin(6 * np.pi * Xnl[:, 0]) + rng.normal(0, 0.1, 300)

lin = LinearRegression()
tree = DecisionTreeRegressor()
forest = RandomForestRegressor(n_estimators=50, random_state=5)

for name, m in [("Linéaire", lin), ("Arbre", tree), ("Forêt", forest)]:
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    scores = cross_val_score(m, Xnl, ynl, cv=kf, scoring='r2')
    print(f"{name:8s} : R² moyen = {scores.mean():.3f} "
          f"(±{scores.std():.3f})")

# Le modèle linéaire est SUREMENT sous-apprenant sur du sin ; la forêt capture.
print("-> Le modèle trop simple (sous-apprend) vs l'arbre très flexible")
print("   (risque de sur-apprendre). La forêt = compromis biais/variance.")


def learning_curve_plot(name, m, X, y):
    """Courbe d'apprentissage : train vs validation sur tailles croissantes."""
    lc = learning_curve(m, X, y, cv=5,
                        train_sizes=np.linspace(0.1, 1.0, 8),
                        scoring='r2', random_state=5)
    train_s, val_s = lc[1], lc[2]  # lc = (train_sizes, train_scores, val_scores)
    print(f"  [{name}] R² train (min->max) = "
          f"{train_s.mean(axis=1)[0]:.3f} -> {train_s.mean(axis=1)[-1]:.3f}")
    print(f"  [{name}] R² val   (min->max) = "
          f"{val_s.mean(axis=1)[0]:.3f} -> {val_s.mean(axis=1)[-1]:.3f}")
    gap = (train_s - val_s).mean(axis=1)
    print(f"  [{name}] écart train-val (indicateur de sur-apprentissage) "
          f"croissant? {gap[-1] > gap[0]}")
    return train_s, val_s, gap


# Arbre profond : sur-apprend (train >> val)
deep = DecisionTreeRegressor(max_depth=None)
learning_curve_plot("Arbre profond (sur-apprend)", deep, Xnl, ynl)
# Arbre limité : meilleur compromis
shallow = DecisionTreeRegressor(max_depth=6)
learning_curve_plot("Arbre limité (meilleur)", shallow, Xnl, ynl)


# ----------------------------------------------------------------------
# 3) SUR-APPRENTISSAGE EXPLICITE (biais-variance)
# ----------------------------------------------------------------------
sep("3. BIAIS-VARIANCE (polygone vs lissé)")
# Données : x = bruit + tendance lissée
x = rng.uniform(-3, 3, 100)
y_true = np.cos(x)
y = y_true + rng.normal(0, 0.1, 100)
Xpoly = x.reshape(-1, 1)
Xtr, Xte, ytr, yte = train_test_split(Xpoly, y, test_size=0.3, random_state=5)

print("Degré du polynôme -> R² train vs R² test :")
for deg in [1, 3, 8, 20]:
    poly = PolynomialFeatures(deg)
    Ptr = poly.fit_transform(Xtr); Pte = poly.transform(Xte)
    m = LinearRegression().fit(Ptr, ytr)
    r2_tr = m.score(Ptr, ytr)
    r2_te = m.score(Pte, yte)
    print(f"  degré {deg:2d} : R²_train={r2_tr:.3f}  R²_test={r2_te:.3f}"
          f"  {'<-- sur-apprend' if r2_tr - r2_te > 0.3 else ''}")

print("-> Bas degré : sous-apprend (biais élevé). Haut degré :")
print("   sur-apprend (variance élevée). Le bon degré = compromis.")


# ----------------------------------------------------------------------
# 4) CLASSIFICATION : MÉTRIQUES
# ----------------------------------------------------------------------
sep("4. CLASSIFICATION (métriques + AUC)")
# Données binaires (2 classes, 1 légèrement déséquilibrée)
Xc = rng.normal(0, 1, (400, 2))
Xc[:200] += np.array([2, 0])      # classe 1
Xc[200:] -= np.array([2, 0])      # classe 0
yc = np.array([1] * 200 + [0] * 200)

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3,
                                              random_state=5)
sc = StandardScaler().fit(Xc_tr)
clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xc_tr), yc_tr)
pred = clf.predict(sc.transform(Xc_te))
prob = clf.predict_proba(sc.transform(Xc_te))[:, 1]

acc = accuracy_score(yc_te, pred)
prec = precision_score(yc_te, pred)
rec = recall_score(yc_te, pred)
f1 = f1_score(yc_te, pred)
auc = roc_auc_score(yc_te, prob)
print(f"Accuracy = {acc:.3f}  Précision = {prec:.3f}  Rappel = {rec:.3f}")
print(f"F1 = {f1:.3f}  AUC = {auc:.3f}")
assert acc > 0.95 and auc > 0.99

# Déséquilibre : 90% classe 0, 10% classe 1
Xd = rng.normal(0, 1, (1000, 2)); yd = np.zeros(1000, int)
Xd[:100] += np.array([3, 0]); yd[:100] = 1
Xdt, Xde, ydt, yde = train_test_split(Xd, yd, test_size=0.3, random_state=5)
scd = StandardScaler().fit(Xdt)
clf2 = LogisticRegression(max_iter=1000).fit(scd.transform(Xdt), ydt)
pred2 = clf2.predict(scd.transform(Xde))
# Un classifieur 'naïf' qui prédit toujours 0 aurait accuracy 0.9 mais F1 nul
acc2 = accuracy_score(yde, pred2)
f1d = f1_score(yde, pred2, zero_division=0)
print(f"\nDonnées déséquilibrées : Accuracy={acc2:.3f} (trompeuse !) "
      f"mais F1={f1d:.3f}")
print("-> En classe déséquilibrée, préférer F1 / AUC à l'accuracy.")


# ----------------------------------------------------------------------
# 5) BIAIS & CONSIDÉRATIONS ÉTHIQUES
# ----------------------------------------------------------------------
sep("5. BIAIS & ÉTHIQUE (détection d'un biais de données)")
# Biais de sélection : l'échantillon de test ne couvre que x>0
# alors que la population inclut x<0
Xfull = rng.uniform(-5, 5, 2000)
yfull = np.sin(Xfull) + rng.normal(0, 0.1, 2000)
# Modélisation faite sur un sous-ensemble biaisé (x>0 seulement)
mask = Xfull > 0
Xb, yb = Xfull[mask].reshape(-1, 1), yfull[mask]
mb = LinearRegression().fit(Xb, yb)
# Le modèle 'croit' en lignes droites -> biais dans l'extrapolation à x<0
print(f"Modèle biaisé (entraîné sur x>0 seulement) : coef={mb.coef_[0]:.3f}")
print("Erreur sur la partie non vue (x<0) :")
err_biais = np.mean((mb.predict(Xfull[~mask].reshape(-1, 1))
                     - yfull[~mask]) ** 2)
print(f"  RMSE sur x<0 = {np.sqrt(err_biais):.3f}")
print("Leçon : un échantillon non représentatif = biais systématique")
print("dans les prédictions, et risque de décisions injustes en inférence.")


# ----------------------------------------------------------------------
# 6) RÉCAPITULATIF : choisir la bonne méthode
# ----------------------------------------------------------------------
sep("6. TABLEAU RÉCAPITULATIF")
print("""
SITUATION                          -> MÉTHODE
------------------------------------------------------------
n grande, x ~ normale              -> tests paramétriques (z, t)
n petit / données non normales     -> tests non paramétriques
3+ groupes (moyennes)              -> ANOVA (+ Tukey post-hoc)
indépendance (catégories)          -> chi-deux
prédiction continue                -> régression (lin, arbre, forêt)
prédiction binaire                 -> régression logistique / arbres
données déséquilibrées             -> F1, AUC (pas l'accuracy)
sur-apprentissage                  -> régularisation, plus de données
biais d'échantillonnage            -> stratification, vérif couverture
""")

print("\nTous les exercices du T07 sont vérifiés. ✅")
