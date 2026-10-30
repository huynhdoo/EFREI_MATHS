#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIF72 — T02 · GÉOMÉTRIE ANALYTIQUE
====================================
- Normes (L1, L2, L∞, Minkowski)
- Produit scalaire, cosinus, angles
- Orthogonalité et projection
- Distances (Euclidienne, Manhattan)
- Bases orthonormales + orthogonalisation de Gram-Schmidt

Lancer :  python3 code/02_geometrie_analytique.py
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=100)


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ----------------------------------------------------------------------
# 1) NORMES
# ----------------------------------------------------------------------
sep("1. NORMES")
x = np.array([3.0, -4.0, 2.0])
print("x =", x)
n1 = np.linalg.norm(x, ord=1)
n2 = np.linalg.norm(x, ord=2)
ninf = np.linalg.norm(x, ord=np.inf)
nmink = np.linalg.norm(x, ord=4)  # Minkowski p=4
print(f"||x||_1 (L1/Manhattan)      = {n1}")
print(f"||x||_2 (L2/Euclidienne)    = {n2}")
print(f"||x||_inf (L∞/max)         = {ninf}")
print(f"||x||_4 (Minkowski p=4)     = {nmink}")
assert n1 == 9 and n2 == np.sqrt(29) and ninf == 4

# Matrice : norme opérateur
M = np.array([[1, 2], [3, 4]])
print("\nNormes de la matrice M :", {
    "L1 (max som lignes? col)": np.linalg.norm(M, ord=1),
    "L2 (singular max)": np.linalg.norm(M, ord=2),
    "Linf": np.linalg.norm(M, ord=np.inf)})


# ----------------------------------------------------------------------
# 2) PRODUIT SCALAIRE, COSINUS, ANGLE
# ----------------------------------------------------------------------
sep("2. PRODUIT SCALAIRE & ANGLE")
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, -1.0, 2.0])
dot = u @ v
cos = dot / (np.linalg.norm(u) * np.linalg.norm(v))
angle = np.arccos(np.clip(cos, -1, 1))
print(f"u·v = {dot}")
print(f"cos(angle) = {cos:.4f}")
print(f"angle = {np.degrees(angle):.2f}°")
assert np.isclose(dot, 1 * 4 + 2 * (-1) + 3 * 2)


# ----------------------------------------------------------------------
# 3) ORTHOGONALITÉ & PROJECTION
# ----------------------------------------------------------------------
sep("3. ORTHOGONALITÉ & PROJECTION")
# projeter u sur v : proj_v(u) = (u·v)/(v·v) * v
def proj(u, v):
    return (u @ v) / (v @ v) * v


u2 = np.array([2.0, 1.0])
v2 = np.array([1.0, 0.0])
p = proj(u2, v2)
print(f"u = {u2}, v = {v2}")
print(f"proj_v(u) = {p}")
print("composante perpendiculaire = u - proj =", u2 - p)
assert np.allclose(p, [2, 0])
assert np.allclose((u2 - p) @ v2, 0)  # perpendiculaire à v


# ----------------------------------------------------------------------
# 4) DISTANCES ENTRE DEUX POINTS
# ----------------------------------------------------------------------
sep("4. DISTANCES")
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 6.0, 3.0])
d_eucl = np.linalg.norm(a - b)
d_manh = np.abs(a - b).sum()
d_cheb = np.max(np.abs(a - b))
print(f"dist euclidienne  = {d_eucl}")
print(f"dist manhattan    = {d_manh}")
print(f"dist chebychev    = {d_cheb}")
assert d_eucl == 5 and d_manh == 7 and d_cheb == 4

# distance de corrélation (utilisée en bioinfo/finance)
def dist_correlation(x, y):
    c = np.corrcoef(x, y)[0, 1]
    return 1 - c


xs = np.array([1.0, 2, 3, 4, 5])
ys = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
print(f"dist de corrélation = {dist_correlation(xs, ys):.5f} "
      "(0 = parfaitement corrélées)")


# ----------------------------------------------------------------------
# 5) BASES ORTHONORMALES — GRAM-SCHMIDT
# ----------------------------------------------------------------------
sep("5. GRAM-SCHMIDT (base orthonormale)")
def gram_schmidt(V):
    """Orthonormalise les colonnes de V."""
    Q = []
    for v in V.T:
        q = v.copy()
        for w in Q:
            q = q - (q @ w) * w
        n = np.linalg.norm(q)
        if n < 1e-12:
            continue
        q = q / n
        Q.append(q)
    return np.stack(Q).T


W = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
Qm = gram_schmidt(W)
print("Colonnes d'origine =\n", W)
print("Base orthonormale Q =\n", Qm)
print("Q.T @ Q ~ Identité :", np.allclose(Qm.T @ Qm, np.eye(2), atol=1e-8))
assert np.allclose(Qm.T @ Qm, np.eye(2), atol=1e-8)


# ----------------------------------------------------------------------
# 6) APPLICATION : similarité de documents (TF · cosinus)
# ----------------------------------------------------------------------
sep("6. APPLICATION — SIMILARITÉ COSINUS")
def cosine_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


# vecteurs de fréquence de mots simplifiés
doc1 = np.array([3, 1, 2, 0])
doc2 = np.array([2, 1, 3, 0])
doc3 = np.array([0, 0, 5, 4])
print(f"sim(doc1,doc2) = {cosine_sim(doc1, doc2):.3f}")
print(f"sim(doc1,doc3) = {cosine_sim(doc1, doc3):.3f}")
assert cosine_sim(doc1, doc2) > cosine_sim(doc1, doc3)


# ======================================================================
# EXERCICES CORRIGÉS
# ======================================================================
sep("EXERCICES CORRIGÉS")

# ---- EXERCICE 1 (CORRIGÉ) : norme -------------------------------------
# Normes L1, L2, L∞ de a=(5, -1, 2)
a1 = np.array([5.0, -1.0, 2.0])
print(f"\nEx1 : L1={np.linalg.norm(a1,1)}, L2={np.linalg.norm(a1,2):.3f}, "
      f"Linf={np.linalg.norm(a1,np.inf)}")
assert np.linalg.norm(a1, 1) == 8

# ---- EXERCICE 2 (CORRIGÉ) : angle -------------------------------------
# Angle entre p=(1,0,0) et q=(0,1,0)
p = np.array([1.0, 0, 0]); q = np.array([0.0, 1, 0])
ang = np.degrees(np.arccos(p @ q / (np.linalg.norm(p) * np.linalg.norm(q))))
print(f"Ex2 : angle(p,q) = {ang}° (orthogonaux)")
assert ang == 90

# ---- EXERCICE 3 (CORRIGÉ) : projection --------------------------------
# Projeter m=(3,4) sur n=(1,2).
# Coef = m·n / n·n = (3*1 + 4*2) / (1^2 + 2^2) = 11/5 = 2.2
# proj_n(m) = 2.2 * (1,2) = (2.2, 4.4)
m = np.array([3.0, 4.0]); n = np.array([1.0, 2.0])
coef = (m @ n) / (n @ n)
pr = coef * n
print(f"Ex3 : coef = {coef}, proj_n(m) = {pr}")
assert np.isclose(coef, 11 / 5)
assert np.allclose(pr, [2.2, 4.4])
# Vérif : m - proj est perpendiculaire à n
assert np.isclose((m - pr) @ n, 0)

# ---- EXERCICE 4 (CORRIGÉ) : distance ----------------------------------
# Distance euclidienne entre (0,0,0) et (1,1,1)
d = np.linalg.norm(np.array([1.0, 1, 1]))
print(f"Ex4 : dist = {d:.4f} = racine(3)")
assert np.isclose(d, np.sqrt(3))

# ---- EXERCICE 5 (CORRIGÉ) : Gram-Schmidt -----------------------------
# Orthonormaliser les colonnes de [[1,1],[1,0]]
Vs = np.array([[1.0, 1.0], [1.0, 0.0]])
Qs = gram_schmidt(Vs)
print("Ex5 : Q =\n", Qs, " Q.T@Q=\n", Qs.T @ Qs)
assert np.allclose(Qs.T @ Qs, np.eye(2), atol=1e-8)

print("\nTous les exercices du T02 sont vérifiés. ✅")
