#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIF72 — T01 · ALGÈBRE LINÉAIRE
================================
Cœur des mathématiques pour la science des données.
- Systèmes d'équations linéaires et opérations matricielles
- Espaces vectoriels, transformations linéaires
- Rang, base, dimension
- Valeurs propres / vecteurs propres, SVD
- Applications à la représentation et manipulation des données

Lancer :  python3 code/01_algebre_lineaire.py
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=100)


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ----------------------------------------------------------------------
# 1) OPÉRATIONS MATRICIELLES DE BASE
# ----------------------------------------------------------------------
sep("1. OPÉRATIONS MATRICIELLES")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("A =\n", A)
print("B =\n", B)
print("A + B   =\n", A + B)
print("A @ B   =\n", A @ B)
print("A.T     =\n", A.T)
print("det(A)  =", np.linalg.det(A))
print("inv(A)  =\n", np.linalg.inv(A))
print("A @ inv(A) ~ Identité:\n", A @ np.linalg.inv(A))


# ----------------------------------------------------------------------
# 2) SYSTÈMES LINÉAIRES  A x = b
# ----------------------------------------------------------------------
sep("2. SYSTÈMES LINÉAIRES  A x = b")
# 2x + 3y - z = 5
#  x -  y + 2z = 6
# 3x + 2y + 2z = 9
A_sys = np.array([[2, 3, -1], [1, -1, 2], [3, 2, 2]])
b_sys = np.array([5, 6, 9])
x = np.linalg.solve(A_sys, b_sys)
print("Solution x =", x)
print("Vérification A @ x =", A_sys @ x, "(attendu ", b_sys, ")")
assert np.allclose(A_sys @ x, b_sys), "Le système n'est pas résolu"


def est_solution(A, b, x):
    return np.allclose(A @ x, b)


print("est_solution ->", est_solution(A_sys, b_sys, x))


# ----------------------------------------------------------------------
# 3) RANG, BASE, DIMENSION (espaces vectoriels)
# ----------------------------------------------------------------------
sep("3. RANG, BASE, DIMENSION")
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("M =\n", M)
print("rang(M) =", np.linalg.matrix_rank(M),
      " (rang < nb colonnes => vecteurs linéairement dépendants)")
print("M est singulière (det ~ 0) :", np.isclose(np.linalg.det(M), 0))

# Image / noyau
U = np.array([[1, 2, 3, 4],
              [2, 4, 6, 8],
              [1, 1, 1, 2]])
print("\nU =\n", U, " rang(U) =", np.linalg.matrix_rank(U))
# base de l'image (colonnes pivots) et du noyau via SVD
_, S, Vt = np.linalg.svd(U, full_matrices=False)
rank = np.sum(S > 1e-9)
print("Base de l'image (colonne pivots de U) : rang =", rank)
print("Base du noyau (vecteurs de Vt à partir du rang) :")
null_basis = Vt[rank:].T
print(null_basis if null_basis.size else "(noyau trivial)")


# ----------------------------------------------------------------------
# 4) TRANSFORMATIONS LINÉAIRES (rotation)
# ----------------------------------------------------------------------
sep("4. TRANSFORMATION LINÉAIRE : ROTATION")
theta = np.pi / 3  # 60°
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
print("Matrice de rotation (60°) =\n", R)
v = np.array([1.0, 0.0])
print("Rotation de (1,0) ->", R @ v)
print("Norme conservée :", np.isclose(np.linalg.norm(R @ v), np.linalg.norm(v)))
print("R @ R.T ~ Identité (orthogonale) :", np.allclose(R @ R.T, np.eye(2)))


# ----------------------------------------------------------------------
# 5) VALEURS PROPRES / VECTEURS PROPRES
# ----------------------------------------------------------------------
sep("5. VALEURS PROPRES")
# Matrice symétrique : valeurs propres réelles, vecteurs orthogonaux
P = np.array([[2, 1], [1, 2]])
eigs, eigvecs = np.linalg.eigh(P)
print("P =\n", P)
print("Valeurs propres =", eigs)
print("Vecteurs propres (colonnes) =\n", eigvecs)
print("Vérification P @ v_i = lambda_i * v_i :\n", P @ eigvecs[:, 0],
      "=", eigs[0], "* ", eigvecs[:, 0])
assert np.allclose(P @ eigvecs, eigvecs * eigs)


# ----------------------------------------------------------------------
# 6) SVD — décomposition singulière (base du PCA)
# ----------------------------------------------------------------------
sep("6. SVD")
X = np.array([[9, 3, 4],
              [2, 6, 7]])
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print("X =\n", X)
print("Singulaires S =", S)
print("X reconstruite = U @ diag(S) @ Vt =\n",
      U @ np.diag(S) @ Vt)
assert np.allclose(U @ np.diag(S) @ Vt, X)
print("\nInterprétation : les singulars triés = 'importance' des directions")


# ----------------------------------------------------------------------
# 7) APPLICATION : PCA SUR DES DONNÉES (reduction de dimension)
# ----------------------------------------------------------------------
sep("7. APPLICATION — PCA MANUEL (SVD)")
rng = np.random.default_rng(0)
# nuage 2D allongé sur une direction
pts = rng.normal(0, 1, (100, 2)) * np.array([3, 1])
Xp = pts - pts.mean(axis=0)
U, S, Vt = np.linalg.svd(Xp, full_matrices=False)
pc1 = Vt[0]
print("1re composante principale (direction) =", pc1)
proj = Xp @ pc1
# Part de variance expliquée par la PC1 = (s1^2) / (somme s_i^2)
exp = (S[0] ** 2) / np.sum(S ** 2)
print(f"Part de variance expliquée par PC1 = {exp:.3f}")
assert exp > 0.9, "La PC1 devrait expliquer la majorité de la variance"


# ======================================================================
# EXERCICES CORRIGÉS
# ======================================================================
sep("EXERCICES CORRIGÉS")

# ---- EXERCICE 1 (CORRIGÉ) : résoudre un système 3x3 ------------------
# Résoudre :  x + 2y + 3z = 6 ;  2x + 3y + 5z = 9 ;  x + 4y + 7z = 10
A1 = np.array([[1, 2, 3], [2, 3, 5], [1, 4, 7]])
b1 = np.array([6, 9, 10])
sol = np.linalg.solve(A1, b1)
print("\nEx1 : solution =", sol, " (vérif :", A1 @ sol, ")")
assert np.allclose(A1 @ sol, b1)

# ---- EXERCICE 2 (CORRIGÉ) : produit + déterminant + inverse ---------
# Soit M = [[3,1],[2,4]]. Calculer det, inverse, et M^2.
M2 = np.array([[3, 1], [2, 4]])
det = np.linalg.det(M2)
inv = np.linalg.inv(M2)
M2_2 = M2 @ M2
print("Ex2 : det(M)=", det, " inv(M)=\n", inv, " M^2=\n", M2_2)
assert np.isclose(det, 10)
assert np.allclose(M2 @ inv, np.eye(2))

# ---- EXERCICE 3 (CORRIGÉ) : rang et dépendance ----------------------
# Est-ce que { (1,2,3), (2,4,6), (1,1,1) } est une base de R^3 ?
v = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
r = np.linalg.matrix_rank(v)
print("Ex3 : rang =", r, "-> base de R^3 ? ", r == 3,
      " (le vect.2 = 2*vect.1, dépendants)")
assert r < 3

# ---- EXERCICE 4 (CORRIGÉ) : valeurs propres d'une matrice ----------
# Valeurs propres de Q = [[4,1],[1,3]].
Q = np.array([[4, 1], [1, 3]])
e, _ = np.linalg.eigh(Q)
print("Ex4 : valeurs propres de Q =", e)
# somme = trace, produit = det
assert np.isclose(sum(e), np.trace(Q))
assert np.isclose(e[0] * e[1], np.linalg.det(Q))

# ---- EXERCICE 5 (CORRIGÉ) : projection sur un vecteur ---------------
# Projeter u=(3,4) sur v=(1,0) et sur v=(1,1).
u = np.array([3.0, 4.0])
for vv in [np.array([1.0, 0.0]), np.array([1.0, 1.0])]:
    proj = (u @ vv) / (vv @ vv) * vv
    print(f"Ex5 : proj de u sur {vv} = {proj}")
assert np.allclose((u @ np.array([1.0, 0.0])) / np.array([1.0]) * np.array([1.0, 0.0]),
                   [3, 0])

print("\nTous les exercices du T01 sont vérifiés. ✅")
