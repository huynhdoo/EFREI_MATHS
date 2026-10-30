#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIF72 — GRAPHIQUES (visualisations pédagogiques)
==================================================
Génère 8 PNG dans ./outputs/ :
  fig01_tcl.png                 — loi de la moyenne + effet du TCL
  fig02_gd_trajectory.png       — descente de gradient 2D (trajectoire)
  fig03_pca.png                 — nuage + PC1 + projection
  fig04_distributions.png       — 4 distributions en parallèle
  fig05_bayes.png               — évolution du prior Bêta -> posterior
  fig06_ic_coverage.png         — intervalles de confiance, couverture
  fig07_surapprentissage.png    — biais-variance (degré polynôme)
  fig08_learning_curves.png     — courbes d'apprentissage (lin vs forêt)

Lancer :  python3 code/09_graphiques.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ----------------------------------------------------------------------
# FIG 01 — TCL
# ----------------------------------------------------------------------
print("Fig 01 : TCL")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
mu, sigma = 10, 3
n_samps = 5000
for n, ax, title in [(1, axes[0], 'n=1 : distribution des tirages'),
                     (30, axes[1], f'n=30 : distribution des X̄ (≈ N({mu},{sigma**2/30}))')]:
    if n == 1:
        data = rng.normal(mu, sigma, n_samps)
    else:
        data = rng.normal(mu, sigma, (n_samps, n)).mean(axis=1)
    ax.hist(data, bins=40, density=True, alpha=0.75, color='#2878b5')
    if n == 1:
        pdf = stats.norm.pdf(data, mu, sigma)
    else:
        pdf = stats.norm.pdf(data, mu, sigma / np.sqrt(n))
    ax.plot(data, pdf, 'r-', lw=2, label='N')
    ax.set_title(title, fontsize=11)
    ax.legend()
fig.suptitle('Théorème de la Limite Centrale : moyenne de n tirages', y=1.02)
save(fig, 'fig01_tcl.png')


# ----------------------------------------------------------------------
# FIG 02 — Trajectoire de descente de gradient
# ----------------------------------------------------------------------
print("Fig 02 : Trajectoire de gradient descent")
f2 = lambda x, y: (x - 3) ** 2 + (y + 1) ** 2 + 0.3 * x * y
df2 = lambda x, y: np.array([2 * (x - 3) + 0.3 * y, 2 * (y + 1) + 0.3 * x])
xx, yy = np.meshgrid(np.linspace(-4, 6, 200), np.linspace(-6, 4, 200))
zz = f2(xx, yy)
# trajectoire
traj = np.array([[-4.0, 0.0]])
for _ in range(40):
    x, y = traj[-1]
    g = df2(x, y)
    traj = np.vstack([traj, [x - 0.5 * g[0], y - 0.5 * g[1]]])
fig, ax = plt.subplots(figsize=(7, 6))
cont = ax.contourf(xx, yy, zz, levels=20, cmap='viridis', alpha=0.8)
ax.plot(traj[:, 0], traj[:, 1], 'r-o', ms=4, lw=1.5, label='Trajectoire GD')
ax.plot(3, -1, 'r*', ms=15, label='Optimum (3,-1)')
ax.plot(traj[0, 0], traj[0, 1], 'wo', ms=8, label='Point de départ')
fig.colorbar(cont, ax=ax, label='f(x,y)')
ax.set_title('Descente de gradient sur f(x,y) = (x-3)²+(y+1)²+0.3xy')
ax.legend(loc='upper left')
save(fig, 'fig02_gd_trajectory.png')


# ----------------------------------------------------------------------
# FIG 03 — PCA
# ----------------------------------------------------------------------
print("Fig 03 : PCA")
pts = rng.normal(0, 1, (300, 2)) * np.array([4, 1.2])
Xc = pts - pts.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
pc1, pc2 = Vt[0], Vt[1]
proj1 = Xc @ pc1
proj2 = Xc @ pc2
var_expl = (S[0] ** 2) / np.sum(S ** 2) * 100
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.4, color='#2878b5', label='Données')
ax.quiver(0, 0, pc1[0] * 8, pc1[1] * 8, angles='xy', scale_units='xy',
          scale=1, color='red', lw=2, label=f'PC1 ({var_expl:.1f}% var)')
ax.quiver(0, 0, pc2[0] * 4, pc2[1] * 4, angles='xy', scale_units='xy',
          scale=1, color='orange', lw=2, label='PC2')
ax.scatter(proj1 * pc1[0], proj1 * pc1[1], s=3, alpha=0.15, color='red',
           label='Projection sur PC1')
ax.set_title(f'PCA manuel (SVD) : PC1 explique {var_expl:.1f}% de la variance')
ax.legend()
ax.set_aspect('equal')
save(fig, 'fig03_pca.png')


# ----------------------------------------------------------------------
# FIG 04 — Distributions
# ----------------------------------------------------------------------
print("Fig 04 : Distributions")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
x = np.linspace(-4, 4, 300)
axes[0, 0].plot(x, stats.norm.pdf(x), color='#2878b5')
axes[0, 0].axvline(0, color='r', ls='--', label='μ=0')
axes[0, 0].set_title('Normale N(0,1)')
axes[0, 0].legend()
k = np.arange(0, 11)
axes[0, 1].bar(k, stats.binom.pmf(k, 10, 0.4), color='#756bb1')
axes[0, 1].set_title('Binomiale n=10, p=0.4')
axes[1, 0].plot(x, stats.expon.pdf(x, scale=1), color='#c12526')
axes[1, 0].set_xlim(0, 6)
axes[1, 0].set_title('Exponentielle λ=1')
axes[1, 1].plot(x, stats.beta.pdf(x, 2, 5), color='#31a354')
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_title('Bêta(2,5)')
fig.suptitle('Lois usuelles en science des données', y=1.02)
save(fig, 'fig04_distributions.png')


# ----------------------------------------------------------------------
# FIG 05 — Bayes (évolution Bêta)
# ----------------------------------------------------------------------
print("Fig 05 : Bayes Bêta-Binomiale")
fig, ax = plt.subplots(figsize=(7, 5))
x = np.linspace(0, 1, 300)
ax.plot(x, stats.beta.pdf(x, 1, 1), lw=2, label='Prior Bêta(1,1)')
for s, t, alpha, col in [(5, 100, 0.5, '#2878b5'),
                          (80, 100, 0.8, '#c12526'),
                          (50, 100, 0.5, '#31a354')]:
    ax.plot(x, stats.beta.pdf(x, 1 + s, 1 + t - s), lw=2,
            label=f'Posterior Bêta({1+s},{1+t-s}) — {s} succès / {t} essais')
ax.set_xlim(0, 1)
ax.set_xlabel('p (probabilité de succès)')
ax.set_ylabel('Densité')
ax.set_title('Mise à jour bayésienne : prior -> posterior')
ax.legend(fontsize=9)
save(fig, 'fig05_bayes.png')


# ----------------------------------------------------------------------
# FIG 06 — Couverture des intervalles de confiance
# ----------------------------------------------------------------------
print("Fig 06 : Couverture IC")
mu_true, sigma_true = 0, 1
N_reps = 2000
fig, ax = plt.subplots(figsize=(8, 5))
for n in [10, 30, 100]:
    ic_means = []
    ic_half = []
    for _ in range(N_reps):
        s = rng.normal(mu_true, sigma_true, n)
        mb = s.mean(); ss = s.std(ddof=1)
        tcrit = stats.t.ppf(0.975, n - 1)
        ic_means.append(mb); ic_half.append(tcrit * ss / np.sqrt(n))
    ic_means = np.array(ic_means); ic_half = np.array(ic_half)
    cov = np.mean((ic_means - ic_half <= mu_true) & (ic_means + ic_half >= mu_true))
    ax.plot(ic_means, ic_means + ic_half, '|', color='#2878b5', alpha=0.15,
            ms=3)
    ax.plot(ic_means, ic_means - ic_half, '|', color='#2878b5', alpha=0.15,
            ms=3)
    print(f"  n={n} : couverture = {cov:.4f} (≈ 95%)")
ax.axhline(0, color='r', lw=2, label='μ vrai = 0')
ax.plot([], [], '|', color='#2878b5', alpha=0.3,
        label=f'IC 95% pour n=10,30,100 (moyennes)')
ax.set_xlabel('Moyenne de l\'échantillon X̄')
ax.set_ylabel('Valeur')
ax.set_title('Couverture des intervalles de confiance 95% (moyenne, t de Student)')
ax.legend()
save(fig, 'fig06_ic_coverage.png')


# ----------------------------------------------------------------------
# FIG 07 — Biais-variance (degré polynôme)
# ----------------------------------------------------------------------
print("Fig 07 : Biais-variance")
xp = rng.uniform(-3, 3, 200)
yp = np.cos(xp) + rng.normal(0, 0.1, 200)
Xp = xp.reshape(-1, 1)
fig, ax = plt.subplots(figsize=(8, 5))
xx = np.linspace(-3, 3, 300).reshape(-1, 1)
ax.plot(xp, yp, 'o', ms=4, alpha=0.5, color='#2878b5', label='Données')
ax.plot(xx, np.cos(xx), 'k-', lw=2, label='Vérité (cos)')
train_r2, test_r2 = [], []
for deg in range(0, 16):
    poly = PolynomialFeatures(deg)
    P = poly.fit_transform(Xp)
    m = LinearRegression().fit(P, yp)
    Pxx = poly.transform(xx)
    r2_tr = m.score(P, yp)
    r2_te = 1 - np.sum((m.predict(Pxx) - np.cos(xx)) ** 2) / np.sum((np.cos(xx) - np.cos(xx).mean()) ** 2)
    train_r2.append(r2_tr); test_r2.append(r2_te)
    ax.plot(xx, m.predict(Pxx), '-', lw=1, alpha=0.6, label=f'degré {deg}')
ax.set_title('Sur-apprentissage : plus le polynôme est riche, plus il colle au bruit')
ax.legend(loc='lower right', fontsize=8, ncols=2)
save(fig, 'fig07_surapprentissage.png')


# ----------------------------------------------------------------------
# FIG 08 — Learning curves
# ----------------------------------------------------------------------
print("Fig 08 : Learning curves")
Xnl = rng.uniform(0, 1, (400, 1))
ynl = np.sin(6 * np.pi * Xnl[:, 0]) + rng.normal(0, 0.1, 400)
fig, ax = plt.subplots(figsize=(8, 5))
for name, m, col in [('Linéaire (sous-apprend)', LinearRegression(), '#31a354'),
                     ('Forêt (bon compromis)',
                      RandomForestRegressor(n_estimators=50, random_state=7),
                      '#2878b5')]:
    lc = learning_curve(m, Xnl, ynl, cv=5,
                        train_sizes=np.linspace(0.1, 1.0, 8),
                        scoring='r2', random_state=7)
    ts = lc[1].mean(axis=1)
    vs = lc[2].mean(axis=1)
    tse = lc[1].std(axis=1)
    vse = lc[2].std(axis=1)
    ax.plot(lc[0], ts, 'o-', color=col, lw=2, label=f'{name} — train')
    ax.plot(lc[0], vs, 's-', color=col, lw=2, ls='--', label=f'{name} — validation')
    ax.fill_between(lc[0], ts - tse, ts + tse, alpha=0.15, color=col)
    ax.fill_between(lc[0], vs - vse, vs + vse, alpha=0.15, color=col)
ax.set_xlabel('Taille de l\'entraînement')
ax.set_ylabel('R²')
ax.set_title('Learning curves : écart train/validation')
ax.legend(fontsize=9)
save(fig, 'fig08_learning_curves.png')


print("\n8 graphiques générés dans ./outputs/. ✅")
