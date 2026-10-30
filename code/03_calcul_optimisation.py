#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIF72 — T03 · CALCUL DIFFÉRENTIEL & OPTIMISATION
===================================================
- Dérivées et règles de base (produit, chaîne)
- Gradient (vecteur des dérivées partielles)
- Descente de gradient (implémentée à la main)
- Optimisation convexe (exemples, Hessian)
- Application ML : régression linéaire et logistique (perte, gradient)

Lancer :  python3 code/03_calcul_optimisation.py
"""
import numpy as np
from scipy import optimize

np.set_printoptions(precision=4, suppress=True, linewidth=100)


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ----------------------------------------------------------------------
# 1) DÉRIVÉES : analytique vs numérique
# ----------------------------------------------------------------------
sep("1. DÉRIVÉES")

# f(x) = x^3  ->  f'(x) = 3x^2
def f(x):
    return x ** 3


def df(x):
    return 3 * x ** 2


# Dérivée numérique (différence centrale, précision O(h^2))
def derive_numerique(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


for xv in [0, 1, 2, 3]:
    print(f"x={xv}:  f' analytic={df(xv):.4f}   numérique={derive_numerique(f, xv):.4f}")
assert np.allclose([df(xv) for xv in [0, 1, 2]],
                   [derive_numerique(f, xv) for xv in [0, 1, 2]])

# Règle de la chaîne : g(x) = sin(x^2)  ->  g'(x) = cos(x^2) * 2x
x = 0.7
g_prime = np.cos(x ** 2) * 2 * x
print(f"\nRègle chaîne : g'(0.7) = cos(0.49)*1.4 = {g_prime:.4f}")
print(f"Vérif numé  : {derive_numerique(lambda t: np.sin(t**2), x):.4f}")
assert np.isclose(g_prime, derive_numerique(lambda t: np.sin(t ** 2), x))


# ----------------------------------------------------------------------
# 2) GRADIENT d'une fonction de plusieurs variables
# ----------------------------------------------------------------------
sep("2. GRADIENT")
# f(x,y) = x^2 + 3xy + y^2
# df/dx = 2x + 3y ;  df/dy = 3x + 2y
def grad_f(x, y):
    return np.array([2 * x + 3 * y, 3 * x + 2 * y])


xy = (1.0, 2.0)
g = grad_f(*xy)
print(f"∇f(1,2) = {g}")
# Vérif par différences
h = 1e-5
num = np.array([
    ( (xy[0]+h)**2 + 3*(xy[0]+h)*xy[1] + xy[1]**2 -
      (xy[0]-h)**2 - 3*(xy[0]-h)*xy[1] - xy[1]**2 ) / (2*h),
    ( xy[0]**2 + 3*xy[0]*(xy[1]+h) + (xy[1]+h)**2 -
      xy[0]**2 - 3*xy[0]*(xy[1]-h) - (xy[1]-h)**2 ) / (2*h)])
print(f"numé      = {num}")
assert np.allclose(g, num, atol=1e-3)


# ----------------------------------------------------------------------
# 3) DESCENTE DE GRADIENT (à la main)
# ----------------------------------------------------------------------
sep("3. DESCENTE DE GRADIENT")
# Minimiser f(x) = (x-3)^2 + (x+1)^2  =>  minimum en x=1, f(1)=8
def f1(x):
    return (x - 3) ** 2 + (x + 1) ** 2


def df1(x):
    return 2 * (x - 3) + 2 * (x + 1)  # = 4x - 4


def gradient_descent(f, df, x0, lr=0.1, n_iter=100, tol=1e-8):
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for i in range(n_iter):
        g = df(x)
        x = x - lr * g
        hist.append(x.copy())
        if np.linalg.norm(g) < tol:
            print(f"  convergé à l'itération {i+1} (|grad|<{tol})")
            break
    return np.array(hist)


hist = gradient_descent(f1, df1, x0=-5.0, lr=0.1, n_iter=200)
print(f"Point de départ  : x0 = -5.0,  f(x0) = {f1(-5):.2f}")
print(f"Dernier point    : x* = {hist[-1]:.4f},  f(x*) = {f1(hist[-1]):.4f}")
print(f"Optimum théorique: x* = 1.0,  f(1) = 8.0")
assert np.isclose(hist[-1], 1.0, atol=1e-3)


# ----------------------------------------------------------------------
# 4) OPTIMISATION CONVEXE (scipy + Hessian)
# ----------------------------------------------------------------------
sep("4. CONVEXITÉ & HESSIAN")
# g(x,y) = x^2 + y^2 - 2x - 4y + 10  (convexe : Hessian = 2I > 0)
H = np.array([[2, 0], [0, 2]])
print("Hessian de g :\n", H)
print("Valeurs propres du Hessian :", np.linalg.eigvalsh(H),
      "(toutes > 0 => strictement convexe)")
res = optimize.minimize(lambda v: v[0]**2 + v[1]**2 - 2*v[0] - 4*v[1] + 10,
                        x0=[0., 0.], jac=lambda v: [2*v[0]-2, 2*v[1]-4])
print("Minimum trouvé (scipy) :", res.x, " valeur :", res.fun)
assert np.allclose(res.x, [1, 2]) and np.isclose(res.fun, 5)


# ----------------------------------------------------------------------
# 5) APPLICATION ML — RÉGRESSION LINÉAIRE par descente de gradient
# ----------------------------------------------------------------------
sep("5. RÉGRESSION LINÉAIRE (GD manuelle)")
rng = np.random.default_rng(42)
n = 200
X = rng.uniform(-5, 5, n)                  # vecteur 1D de taille n
theta_true = 2.5
b_true = 1.0
noise = rng.normal(0, 0.5, n)
y = theta_true * X + b_true + noise


def loss_mse(theta, b, X, y):
    pred = theta * X + b
    return np.mean((pred - y) ** 2)


def grad_mse(theta, b, X, y):
    pred = theta * X + b
    err = pred - y
    g_theta = (2 / len(y)) * np.sum(X * err)
    g_b = (2 / len(y)) * np.sum(err)
    return g_theta, g_b


theta, b = 0.0, 0.0
lr = 0.01
print(f"Départ : theta={theta}, b={b}, loss={loss_mse(theta, b, X, y):.3f}")
for it in range(2000):
    gt, gb = grad_mse(theta, b, X, y)
    theta -= lr * gt
    b -= lr * gb
print(f"Fin    : theta={theta:.3f} (vrai={theta_true}), "
      f"b={b:.3f} (vrai={b_true}), loss={loss_mse(theta, b, X, y):.4f}")
assert abs(theta - theta_true) < 0.1 and abs(b - b_true) < 0.1


# ----------------------------------------------------------------------
# 6) APPLICATION ML — RÉGRESSION LOGISTIQUE (GD)
# ----------------------------------------------------------------------
sep("6. RÉGRESSION LOGISTIQUE (GD manuelle)")
# Génération de données binaires
X2 = rng.normal(0, 1, (n, 2))
w_true = np.array([1.5, -2.0])
p = 1 / (1 + np.exp(-(X2 @ w_true + 0.5)))
y2 = (rng.uniform(size=n) < p).astype(int)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def log_loss(w, b, X, y):
    z = sigmoid(X @ w + b)
    eps = 1e-12
    return -np.mean(y * np.log(z + eps) + (1 - y) * np.log(1 - z + eps))


def log_loss_grad(w, b, X, y):
    z = sigmoid(X @ w + b)
    err = z - y
    gw = (1 / len(y)) * (X.T @ err)
    gb = (1 / len(y)) * np.sum(err)
    return gw, gb


w, b = np.zeros(2), 0.0
for it in range(2000):
    gw, gb = log_loss_grad(w, b, X2, y2)
    w -= 0.1 * gw
    b -= 0.1 * gb
acc = np.mean((sigmoid(X2 @ w + b) >= 0.5).astype(int) == y2)
print(f"Vrais poids = {w_true},  estimés = {w}")
print(f"Accuracy = {acc:.3f}   (perdita = {log_loss(w, b, X2, y2):.4f})")
assert acc > 0.7 and abs(w[0] - w_true[0]) < 0.4 and abs(w[1] - w_true[1]) < 0.4

# Comparaison avec sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr_model = LogisticRegression(max_iter=1000).fit(X2, y2)
print(f"sklearn : weights={lr_model.coef_[0]}, accuracy="
      f"{accuracy_score(y2, lr_model.predict(X2)):.3f}")

print("\nTous les exercices du T03 sont vérifiés. ✅")
