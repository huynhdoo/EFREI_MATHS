#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADIF72 — T06 · INFÉRENCE STATISTIQUE & TESTS
=============================================
- Tests paramétriques (z, t, binomial)
- Tests non paramétriques (Mann-Whitney, Wilcoxon, Kruskal)
- ANOVA (comparaison de 3+ groupes)
- Test du chi-deux (indépendance)
- Statistique bayésienne (conjugaison Bêta-Binomiale)
- Théorie de la décision (risque, coût de décision)

Lancer :  python3 code/06_tests_et_inference.py
"""
import numpy as np
from scipy import stats

np.set_printoptions(precision=4, suppress=True, linewidth=100)


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ----------------------------------------------------------------------
# 1) TESTS PARAMÉTRIQUES
# ----------------------------------------------------------------------
sep("1. TESTS PARAMÉTRIQUES")
rng = np.random.default_rng(3)

# 1a) Test z pour une moyenne (variance connue)
# H0: mu=0 vs H1: mu≠0, échantillon N(0.3, 1), n=100
n = 100
sample_z = rng.normal(0.3, 1.0, n)
z = (sample_z.mean() - 0) / (1.0 / np.sqrt(n))
p_val = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"1a) z-test : z={z:.3f}, p={p_val:.4f} -> "
      f"{'rejeter' if p_val<0.05 else 'non'} H0")
assert p_val < 0.05

# 1b) Test t pour deux échantillons (indépendants)
g1 = rng.normal(10, 1, 40); g2 = rng.normal(10.8, 1, 40)
t, p_t = stats.ttest_ind(g1, g2, equal_var=True)
print(f"1b) t-test : t={t:.3f}, p={p_t:.4f} -> "
      f"{'différent' if p_t<0.05 else 'pas de diff'}")
assert p_t < 0.05

# 1c) Test du chi (indépendance sur effectifs)
# Tableau 2x3 : 2 catégories x 3 niveaux
table = np.array([[20, 30, 45],
                  [30, 25, 35]])
chi2, p_chi, dof, _ = stats.chi2_contingency(table)
print(f"1c) chi2-indépendance : chi2={chi2:.3f}, dof={dof}, "
      f"p={p_chi:.4f} -> {'dépendants' if p_chi<0.05 else 'indépendants'}")

# 1d) Test binomial (proportion)
# 100 essais, 45 succès, H0: p=0.5
stat_b, p_b = None, stats.binomtest(45, 100, 0.5, alternative='two-sided').pvalue
print(f"1d) binomial : p={p_b:.4f} -> "
      f"{'rejet' if p_b<0.05 else 'non'} de p=0.5")


# ----------------------------------------------------------------------
# 2) TESTS NON PARAMÉTRIQUES (sans hypothèse de normalité)
# ----------------------------------------------------------------------
sep("2. TESTS NON PARAMÉTRIQUES")
# 2a) Mann-Whitney U (deux échantillons, médianes)
a = rng.normal(5, 1, 50); b = np.abs(rng.normal(5, 1, 50)) + rng.normal(0,0.1,50)
u, p_u = stats.mannwhitneyu(a, b, alternative='two-sided')
print(f"2a) Mann-Whitney : U={u:.1f}, p={p_u:.4f}")

# 2b) Test de Wilcoxon signé (échantillons appariés)
before = rng.normal(10, 1, 30); after = before + rng.normal(0.5, 0.5, 30)
w, p_w = stats.wilcoxon(before, after, alternative='two-sided')
print(f"2b) Wilcoxon signé : W={w:.1f}, p={p_w:.4f} -> "
      f"{'diff' if p_w<0.05 else 'non diff'}")

# 2c) Kruskal-Wallis (3+ groupes)
gA = rng.normal(0, 1, 30); gB = rng.normal(0.5, 1, 30); gC = rng.normal(1, 1, 30)
k, p_k = stats.kruskal(gA, gB, gC)
print(f"2c) Kruskal-Wallis : H={k:.3f}, p={p_k:.4f} -> "
      f"{'au moins un groupe diff' if p_k<0.05 else 'n/d'}")
assert p_k < 0.05

# 2d) Test de normalité Shapiro (prérequis des tests paramétriques)
norm_s = rng.normal(0, 1, 50)
w_sp, p_sp = stats.shapiro(norm_s)
print(f"2d) Shapiro (normale) : p={p_sp:.3f} (>{0.05} => normale)")
skew_s = rng.exponential(1, 50)
w_sk, p_sk = stats.shapiro(skew_s)
print(f"2d) Shapiro (expo)    : p={p_sk:.3f} (<{0.05} => non normale)")


# ----------------------------------------------------------------------
# 3) ANOVA (comparaison de 3+ moyennes)
# ----------------------------------------------------------------------
sep("3. ANOVA")
# 3 traitements, H0: mu1=mu2=mu3
t1 = rng.normal(20, 2, 25); t2 = rng.normal(20, 2, 25); t3 = rng.normal(24, 2, 25)
f, p_anova = stats.f_oneway(t1, t2, t3)
print(f"ANOVA 1 facteur : F={f:.3f}, p={p_anova:.4f} -> "
      f"{'moyennes diff' if p_anova<0.05 else 'pas de diff'}")
assert p_anova < 0.05

# Test post-hoc : Tukey HSD (via q de Student)
def tukey_pairwise(groups, alpha=0.05):
    """Tukey HSD simplifié pour p groupes (comparaisons par paires)."""
    gs = list(groups); k = len(gs)
    mse = sum(np.var(g, ddof=1) for g in gs) / k
    df = sum(len(g) - 1 for g in gs)
    q = stats.studentized_range.ppf(1 - alpha, k, df)
    hsd = q * np.sqrt(mse / min(len(g) for g in gs))
    res = {}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(gs[i].mean() - gs[j].mean())
            res[f"g{i}-g{j}"] = diff > hsd
    return res, hsd

posthoc, hsd = tukey_pairwise([t1, t2, t3])
print(f"HSD (seuil) = {hsd:.3f}")
for pair, sig in posthoc.items():
    print(f"  {pair}: {'SIGNIFICATIF' if sig else 'n/d'}")
assert posthoc['g0-g2'] and not posthoc['g0-g1']


# ----------------------------------------------------------------------
# 4) TEST DU CHI-DEUX (indépendance)
# ----------------------------------------------------------------------
sep("4. CHI-DEUX — INDÉPENDANCE")
# Croisé : genre (M/F) x satisfaction (Oui/Non/Neutre)
obs = np.array([[45, 30, 25],
                [35, 40, 25]])
chi2, p, dof, exp = stats.chi2_contingency(obs)
print(f"Effectifs observés :\n{obs}")
print(f"chi2={chi2:.3f}, dof={dof}, p={p:.4f} -> "
      f"{'indépendants' if p>0.05 else 'liés'}")


# ----------------------------------------------------------------------
# 5) STATISTIQUE BAYÉSIENNE (Bêta-Binomiale)
# ----------------------------------------------------------------------
sep("5. BAYÉSIENNE — BÊTA-BINOMIALE")
# Prior uniforme Bêta(1,1). 60 succès, 40 échecs -> posterior Bêta(61,41)
alpha, beta = 1, 1
s, e = 60, 40
a_post, b_post = alpha + s, beta + e
# Crédit de proba : P(p > 0.5)
p_gt_05 = 1 - stats.beta.cdf(0.5, a_post, b_post)
print(f"Prior Bêta({alpha},{beta}) + {s}/{s+e} -> "
      f"Posterior Bêta({a_post},{b_post})")
print(f"E[p] = {a_post/(a_post+b_post):.4f} ; Var[p] = "
      f"{a_post*b_post/((a_post+b_post)**2*(a_post+b_post+1)):.5f}")
print(f"P(p > 0.5) = {p_gt_05:.5f}")
assert p_gt_05 > 0.97


# ----------------------------------------------------------------------
# 6) THÉORIE DE LA DÉCISION (coûts)
# ----------------------------------------------------------------------
sep("6. THÉORIE DE LA DÉCISION")
# Matrice de coût (faux négatif coûte plus que faux positif)
#                 Décision: Malade    Décision: Sain
# Réel Malade : [    0 (bon)     ,   100 (FN)  ]
# Réel Sain   : [    20 (FP)     ,    0 (bon)  ]
C = np.array([[0, 100],
              [20, 0]])
p_malade = 0.5
# Coût attendu de chaque décision
cost_dec_malade = p_malade * C[0, 0] + (1 - p_malade) * C[1, 0]
cost_dec_sain = p_malade * C[0, 1] + (1 - p_malade) * C[1, 1]
print(f"Coût moyen 'diagnostiquer malade' = {cost_dec_malade:.1f}")
print(f"Coût moyen 'diagnostiquer sain'    = {cost_dec_sain:.1f}")
print(f"-> Choisir {'malade' if cost_dec_malade < cost_dec_sain else 'sain'} "
      f"(minimise le coût)")
assert cost_dec_malade < cost_dec_sain


# ======================================================================
# EXERCICES CORRIGÉS
# ======================================================================
sep("EXERCICES CORRIGÉS")

# ---- EXERCICE 1 (CORRIGÉ) : test t 1 échantillon ------------------------
# On veut tester si la moyenne est 100. 15 mesures autour de 103.
r = np.random.default_rng(11)
data = r.normal(103, 4, 15)
t, p = stats.ttest_1samp(data, 100)
print(f"\nEx1 : xbar={data.mean():.2f}, t={t:.3f}, p={p:.4f} -> "
      f"{'≠100' if p<0.05 else '=100'}")
assert p < 0.05

# ---- EXERCICE 2 (CORRIGÉ) : test de proportions (z) --------------------
# 200 sondés, 120 favorables. H0: p=0.5.
phat = 120 / 200
z = (phat - 0.5) / np.sqrt(0.5 * 0.5 / 200)
p = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"Ex2 : z={z:.3f}, p={p:.4f} -> "
      f"{'p≠0.5' if p<0.05 else 'p=0.5'}")
assert p < 0.05

# ---- EXERCICE 3 (CORRIGÉ) : ANOVA 1 facteur ----------------------------
r2 = np.random.default_rng(21)
A = r2.normal(0, 1, 20); B = r2.normal(0, 1, 20); C = r2.normal(0, 1, 20)
f, p = stats.f_oneway(A, B, C)
print(f"Ex3 : F={f:.3f}, p={p:.3f} -> "
      f"{'diff' if p<0.05 else 'n/d'} (ici même moyenne, p élevé)")
assert p > 0.05  # groupes de même moyenne -> non significatif

# ---- EXERCICE 4 (CORRIGÉ) : chi2 (conformité) --------------------------
# Effectifs attendus [25,25,25,25] vs observés [30,20,25,25]
obs4 = np.array([30, 20, 25, 25])
chi2, p = stats.chisquare(obs4, [25, 25, 25, 25])
print(f"Ex4 : chi2={chi2:.3f}, p={p:.4f} -> "
      f"{'non conforme' if p<0.05 else 'conforme'}")
assert p > 0.05

# ---- EXERCICE 5 (CORRIGÉ) : test non paramétrique Mann-Whitney ----------
r3 = np.random.default_rng(31)
x = r3.exponential(1, 40); y = r3.exponential(1.5, 40)  # y plus "élevées"
u, p = stats.mannwhitneyu(x, y, alternative='two-sided')
print(f"Ex5 : U={u:.1f}, p={p:.4f} -> "
      f"{'distributions diff' if p<0.05 else 'n/d'}")

print("\nTous les exercices du T06 sont vérifiés. ✅")
