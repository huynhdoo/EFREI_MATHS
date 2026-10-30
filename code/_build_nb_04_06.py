#!/usr/bin/env python3
# Build notebooks 04, 05, 06 for ADIF72
import nbformat as nbf
import os

def md(text):
    return nbf.v4.new_markdown_cell(text)
def code(text):
    return nbf.v4.new_code_cell(text)
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
# NOTEBOOK 04 — THÉORIE DES PROBABILITÉS
# ======================================================================
cells04 = [
md("""# T04 · Théorie des probabilités — ADIF72

**Objectifs :**
- Espace probabiliste et axiomes de Kolmogorov
- Variables aléatoires discrètes et continues
- Lois usuelles : binomiale, poisson, normale, bêta, exponentielle
- Espérance, variance, écart-type
- Théorème de Bayes
- Modélisation d'incertitude (Monte-Carlo, VaR)

**Pourquoi c'est central :** la probabilité est le **pont** entre la
géométrie (T01-T02), le calcul (T03) et la statistique (T05-T06).
Toute inférence repose sur le modèle de probabilité qu'on adopte.""", ),
code("""import numpy as np
from scipy import stats
np.set_printoptions(precision=4, suppress=True, linewidth=100)
print("Prêt. scipy", stats.norm.cdf(0))"""),
md("""## 1 · Espace probabiliste & axiomes de Kolmogorov

Un **espace probabiliste** est un triplet `(Ω, F, P)` :
- `Ω` : ensemble des résultats possibles (espace fondamental)
- `F` : ensemble des événements (σ-algèbre)
- `P` : fonction de probabilité

**Axiomes :**
1. `P(A) ≥ 0` pour tout événement `A`
2. `P(Ω) = 1`
3. Pour des événements **disjoints** `Aᵢ` : `P(∪ Aᵢ) = Σ P(Aᵢ)`

Ces 3 axiomes suffisent à dériver **toute** la théorie classique.
En pratique, on définit directement `P` sur les éléments de base et on
propage par additivité.""", ),
code("""# Dé 6 faces équilibré
sample_space = {i: 1/6 for i in range(1, 7)}
assert np.isclose(sum(sample_space.values()), 1.0), "P(Ω)=1"
assert all(v >= 0 for v in sample_space.values()), "P≥0"
P_impair = sum(sample_space[i] for i in [1, 3, 5])
print(f"P({{1,3,5}}) = {P_impair} = 3/6 (additivité)")
assert np.isclose(P_impair, 0.5)"""),
md("""## 2 · Variable aléatoire discrète

Une **v.a. discrète** `X` prend une valeur finie ou dénombrable. Sa
**distribution** (PMF) : `P(X = k)`.
Conditions : `P(X=k) ≥ 0` et `Σ_k P(X=k) = 1`.
Moments :
```
E[X]   = Σ k · P(X=k)
E[X²]  = Σ k² · P(X=k)
Var[X] = E[X²] - (E[X])²
```
Exemple : nombre de piles sur 4 lancers = **Binomiale(n=4, p=0.5)**.""", ),
code("""n, p = 4, 0.5
k = np.arange(n + 1)
pmf = stats.binom.pmf(k, n, p)
for kv, pv in zip(k, pmf):
    print(f"P(X={kv}) = {pv:.4f}")
print("Somme =", pmf.sum())
print(f"E[X] = np = {n*p:.2f}   Var[X] = np(1-p) = {n*p*(1-p):.2f}")
assert np.isclose(pmf.sum(), 1.0)"""),
md("""## 3 · Variable aléatoire continue — Normal

Une **v.a. continue** `X` a une **densité** `f(x)` (pas une probabilité
directe : on intègre pour obtenir des probas).
`∫ f(x) dx = 1`.
Pour `X ~ N(μ, σ²)` :
- `E[X] = μ`, `Var[X] = σ²`
- `P(a < X < b) = Φ((b-μ)/σ) - Φ((a-μ)/σ)`
- **Règle 68-95-99.7** : `P(|Z|<1)=68%`, `<2)=95%`, `<3)=99.7%`

C'est la distribution **fondamentale** — le TCL (T05) garantit que
presque toute moyenne d'échantillon est quasi normale, même si la
population ne l'est pas.""", ),
code("""mu, sigma = 2.0, 1.5
print(f"X ~ N({mu}, {sigma}²)")
print(f"E[X] = {mu}   Var[X] = {sigma**2}   sd = {sigma}")
c = stats.norm.cdf(3, mu, sigma) - stats.norm.cdf(0, mu, sigma)
print(f"P(0<X<3) = {c:.4f}")
q05 = stats.norm.ppf(0.05, mu, sigma); q95 = stats.norm.ppf(0.95, mu, sigma)
print(f"5e percentile = {q05:.3f}, 95e = {q95:.3f}")
for k_, pct in [(1, 68.27), (2, 95.45), (3, 99.73)]:
    val = 2 * stats.norm.cdf(k_) - 1
    print(f"P(|Z|<{k_}) = {val:.4f}  (~{pct}%)")"""),
md("""## 4 · Lois usuelles en science des données

| Loi | Paramètres | Usage typique |
|---|---|---|
| **Binomiale** | n, p | Succès/échecs (n essais) |
| **Poisson** | λ | Événements rares (appels, défauts) |
| **Normale** | μ, σ² | Données naturelles, moyenne (TCL) |
| **Exponentielle** | λ | Temps entre événements |
| **Bêta** | α, β | Proportions / **prior bayésien** |
| **LogNormale** | μ, σ | Prix, tailles, revenus (>0, asym) |

**Choix de loi** = première décision en modélisation. La loi détermine
les moments (E, Var), les tests statistiques, et l'inférence possible.""", ),
code("""lam = 3.0
print(f"Poisson(λ={lam}) : P(X=5) = {stats.poisson.pmf(5, lam):.4f}  E={lam} Var={lam}")
print(f"Binomiale(n=20,p=0.3) : P(X=5) = {stats.binom.pmf(5,20,0.3):.4f}")
print(f"Bernoulli(p=0.7) : E={0.7}, Var={0.7*0.3:.4f}")
print(f"Exponentielle(λ=2) : P(X<0.5) = {stats.expon.cdf(0.5, scale=0.5):.4f}")
print(f"Bêta(2,5) : E={2/7:.3f}, Var={2*5/(7**2*8):.4f}")"""),
md("""## 5 · Théorème de Bayes

**Formule** : pour des événements `A, B` avec `P(B) > 0` :
```
P(A|B) = P(B|A) P(A) / P(B)
```
- `P(A)` : **prior** (avant observation)
- `P(B|A)` : **vraisemblance** (comment B se présente si A est vrai)
- `P(B)` : **évidence** (normalisation)
- `P(A|B)` : **posterior** (après observation)

### L'exemple du test de dépistage (paradoxe du taux de base)
Un test a 99% de sensibilité et 5% de faux positifs. Prévalence = 1%.
→ Quelle est `P(malade | positif)` ?
Réponse contre-intuitive : **~17%**. La rareté de la maladie domine.
C'est un piège **très fréquent** en examen et en pratique.""", ),
code("""P_malade = 0.01
P_pos_malade = 0.99    # sensibilité
P_pos_sain = 0.05      # faux positifs
P_pos = P_pos_malade * P_malade + P_pos_sain * (1 - P_malade)
P_malade_pos = P_pos_malade * P_malade / P_pos
print(f"P(Positif) = {P_pos:.5f}")
print(f"P(Malade | Positif) = {P_malade_pos:.5f}  ({P_malade_pos*100:.2f}%)")
print("-> malgré 99% de sensibilité, très peu de malades (taux de base)")
assert P_malade_pos < 0.2"""),
md("""### Bayes avec prior Bêta (conjugaison Bêta-Binomiale)
Si `p ~ Bêta(α, β)` et qu'on observe `k` succès sur `n` essais,
alors `p | données ~ Bêta(α+k, β+n-k)`. C'est la forme la plus simple
de l'inférence bayésienne — on la retrouve partout (A/B testing,
modèles binaires).""", ),
code("""a, b = 1, 1   # prior uniforme
successes, trials = 3, 100
a_post, b_post = a + successes, b + trials - successes
print(f"Prior Bêta({a},{b}) + {successes} succès sur {trials}")
print(f"-> Posterior Bêta({a_post},{b_post})")
print(f"   E[prior]={a/(a+b):.3f} -> E[posterior]={a_post/(a_post+b_post):.3f}")
print(f"   Var[posterior]={a_post*b_post/((a_post+b_post)**2*(a_post+b_post+1)):.5f}")"""),
md("""## 6 · Test d'hypothèse (aperçu — détaillé en T06)

Le test répond à : « Les données contredisent-elles l'hypothèse nulle ? »
On calcule un **p-value** = P(observer un résultat aussi extrême
que celui observé **si H0 était vraie**).
- `p < α` (souvent 0.05) → rejeter H0
- `p ≥ α` → pas assez de preuve pour rejeter

Exemple : 60 piles sur 100 lancers. Est-ce une pièce équilibrée ?""", ),
code("""n, k, p0 = 100, 60, 0.5
z = (k - n*p0) / np.sqrt(n*p0*(1-p0))   # approx. normale
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"z = {z:.3f}, p-value = {p_value:.5f}")
alpha = 0.05
print(f"A α={alpha}: {'REJET' if p_value < alpha else 'non-rejet'} de H0 (pièce équilibrée)")
assert p_value < 0.05"""),
md("""## 7 · Modélisation d'incertitude & risque (Monte-Carlo)

**Monte-Carlo** : approximer une intégrale / probabilité par simulation.
Utilisé quand la solution analytique est complexe (modèles financiers,
physiques, réseaux de neurones).
Convergence : l'erreur diminue en `O(1/√N)` — pour diviser par 10 l'erreur,
il faut 100× plus de simulations.""", ),
code("""rng = np.random.default_rng(7)
mu_l, sigma_l = 4.5, 0.5
N = 200_000
X = rng.lognormal(mu_l, sigma_l, N)
p_mc = np.mean(X > 100)
p_exact = 1 - stats.lognorm.cdf(100, s=sigma_l, scale=np.exp(mu_l))
print(f"P(X>100) : Monte-Carlo = {p_mc:.5f}, exacte = {p_exact:.5f}")
assert abs(p_mc - p_exact) < 0.01
var95 = np.percentile(X, 95)
print(f"VaR 95% = {var95:.3f}")"""),
md("""## 🎯 Exercices corrigés
"""),
code("""# Ex 1 : P(face pair ET pile) = (3/6)*(1/2) = 1/4
p_ex1 = (3/6) * (1/2)
print(f"Ex1 : {p_ex1}")
assert p_ex1 == 0.25"""),
code("""# Ex 2 : E[X] et Var[X] d'une v.a. discrète
# X ~ {0,1,2} avec P = {0.2, 0.5, 0.3}
xx = np.array([0, 1, 2]); ppx = np.array([0.2, 0.5, 0.3])
ex = (xx * ppx).sum()
ex2 = ((xx ** 2) * ppx).sum()
var = ex2 - ex ** 2
print(f"Ex2 : E[X]={ex}, E[X²]={ex2}, Var[X]={var}")
assert np.isclose(ex, 1.1) and np.isclose(var, 0.49)"""),
code("""# Ex 3 : P(0 ≤ Z ≤ 1) pour Z ~ N(0,1)
z01 = stats.norm.cdf(1) - stats.norm.cdf(0)
print(f"Ex3 : {z01:.4f}")
assert np.isclose(z01, 0.3413, atol=1e-3)"""),
code("""# Ex 4 : Binomiale n=50, p=0.4 -> P(X=20) et P(X≥40)
n, p = 50, 0.4
p_eq20 = stats.binom.pmf(20, n, p)
p_ge40 = 1 - stats.binom.cdf(39, n, p)
print(f"Ex4 : P(X=20)={p_eq20:.4f}, P(X≥40)={p_ge40:.4f}")
assert 0 < p_eq20 < 1 and p_ge40 < 1e-3"""),
code("""# Ex 5 : Bayes 2 classes
# P(A)=0.6, P(B|A)=0.3, P(B|Aᶜ)=0.8 -> P(A|B)
p_a, p_b_given_a, p_b_given_not = 0.6, 0.3, 0.8
p_b = p_b_given_a * p_a + p_b_given_not * (1 - p_a)
p_a_given_b = p_b_given_a * p_a / p_b
print(f"Ex5 : P(B)={p_b:.3f}, P(A|B)={p_a_given_b:.4f}")
assert np.isclose(p_a_given_b, 0.36)"""),
code("""# Ex 6 : Loi de Poisson λ=2.5, P(X=3)
p3 = stats.poisson.pmf(3, 2.5)
print(f"Ex6 : {p3:.4f}")
assert np.isclose(p3, 0.2138, atol=1e-3)

print("\\n✅ T04 terminé.")"""),
]

# ======================================================================
# NOTEBOOK 05 — MODÉLISATION STATISTIQUE
# ======================================================================
cells05 = [
md("""# T05 · Modélisation statistique — ADIF72

**Objectifs :**
- Distributions d'échantillonnage
- Théorème de la Limite Centrale (TCL)
- Estimation : biais, variance, MSE
- Intervalles de confiance (moyenne, proportion)
- Régression linéaire & corrélation

**Lien avec T04 :** si `X` a une loi (mêlée), la moyenne d'un échantillon
a **aussi** une loi (distribution d'échantillonnage). C'est ce qui permet
d'inférer (T06) et de construire des intervalles de confiance.""", ),
code("""import numpy as np
import pandas as pd
from scipy import stats
import sklearn.linear_model as lm
np.set_printoptions(precision=4, suppress=True, linewidth=100)
print("Prêt. sklearn", lm.LinearRegression.__module__)"""),
md("""## 1 · Distribution d'échantillonnage

Si `X ~ (μ, σ²)`, et `X₁...Xₙ` i.i.d., alors la **moyenne d'échantillon** :
```
X̄ = (1/n) Σ Xᵢ   ~   (μ, σ²/n)
```
C'est-à-dire : `E[X̄] = μ` et `Var(X̄) = σ²/n`.
Conséquence clé : **l'erreur standard diminue en `1/√n`**. Pour
diviser par 2 l'erreur, il faut 4× plus de données. C'est le coût
de la précision statistique.

On peut aussi écrire `X̄ ~ N(μ, σ²/n)` **si** la population est normale ;
sinon, le TCL (suivant) garantit la normalité **asymptotique**.""", ),
code("""mu, sigma, n = 10.0, 3.0, 25
rng = np.random.default_rng(0)
means = [rng.normal(mu, sigma, n).mean() for _ in range(2000)]
means = np.array(means)
print(f"Population N({mu},{sigma}) ; échantillons n={n}")
print(f"Moyenne des X̄ = {means.mean():.4f}  (≈ {mu})")
print(f"SD des X̄      = {means.std(ddof=1):.4f}  (≈ {sigma/np.sqrt(n):.4f})")
assert abs(means.mean() - mu) < 0.1
assert abs(means.std(ddof=1) - sigma/np.sqrt(n)) < 0.1"""),
md("""## 2 · Théorème de la Limite Centrale (TCL)

**Énoncé (simplifié)** : si `X₁,...,Xₙ` sont i.i.d. avec moyenne `μ`
et variance finie `σ²`, alors :
```
Z_n = (X̄ - μ) / (σ/√n)  --(n→∞)-->  N(0,1)
```
**C'est le théorème le plus important de la statistique.** Il dit que :
- La moyenne d'échantillons devient **normale**, peu importe la loi source
- La vitesse de convergence est en `O(1/√n)`

Applications directes :
- **Intervalles de confiance** (section 4)
- **Tests z** (T06)
- Justification du Gaussian noise dans les réseaux de neurones""", ),
code("""# Population uniforme [0,1] (pas normale !) -> la moyenne est quasi normale
n = 30
mu_u, var_u = 0.5, 1/12
rng = np.random.default_rng(0)
Z = [(rng.uniform(0, 1, n).mean() - mu_u) / np.sqrt(var_u / n)
     for _ in range(5000)]
Z = np.array(Z)
print(f"Tirages de la moyenne de {n} uniformes, standardisés")
print(f"  E[Z]  = {Z.mean():.4f}  (attendu 0)")
print(f"  Var[Z]= {Z.var():.4f}  (attendu 1)")
print(f"  kurtosis (excès) = {stats.kurtosis(Z):.4f}  (normale: 0)")
w, p = stats.shapiro(Z[:500])
print(f"  Shapiro-Wilk p = {p:.3f}  (>0.05 => compatible normalité)")
assert abs(Z.mean()) < 0.05 and abs(Z.var() - 1) < 0.1"""),
md("""## 3 · Estimation — biais & variance d'un estimateur

Un **estimateur** est une règle (fonction des données) qui produit une
valeur pour un paramètre inconnu (ici `μ`).

Mesures de qualité :
```
Biais(θ̂) = E[θ̂] - θ
Var(θ̂)   = Var(θ̂)
MSE(θ̂)   = Biais² + Var(θ̂)
```
- **Sans biais** : `E[θ̂] = θ` (pas de dérive moyenne)
- **Efficient** : parmi les estimateurs sans biais, le plus petite variance

L'exemple `moyenne vs 1re observation` montre que deux estimateurs
peuvent être sans biais, mais l'un est **beaucoup plus précis**
(variance `σ²/n` vs `σ²`).""", ),
code("""mu, sigma = 10.0, 3.0
def bias_variance(estimator, n_trials=20000, rng=np.random.default_rng(1)):
    vals = np.array([estimator(rng) for _ in range(n_trials)])
    return vals.mean() - mu, vals.var(), vals

est_mean = lambda r: r.normal(mu, sigma, size=25).mean()
est_first = lambda r: r.normal(mu, sigma, size=1)[0]
bm, vm, _ = bias_variance(est_mean)
bf, vf, _ = bias_variance(est_first)
print(f"moyenne (n=25) : bias={bm:+.4f}, var={vm:.4f}, MSE={bm**2+vm:.4f}")
print(f"1re obs        : bias={bf:+.4f}, var={vf:.4f}, MSE={bf**2+vf:.4f}")
print(f"Var de moyenne / σ² = {vm/sigma**2:.4f} (= 1/n)")
assert abs(bm) < 0.02 and vm < vf"""),
md("""## 4 · Intervalles de confiance (IC)

Un **IC à 95%** pour un paramètre est un intervalle `[a, b]` construit
à partir des données tel que :
- On dit « le paramètre **est** dans `[a,b]` avec 95% de confiance ».
- **Interprétation correcte** : si on répétait l'expérience 100 fois,
  **95** des intervalles produiraient contiendraient le vrai paramètre.
- Le vrai paramètre est **fixé** (pas aléatoire) — c'est l'intervalle
  qui varie. C'est le piège n°1 à l'examen.

**Cas 1** — variance connue : `IC = X̄ ± z₀.₉₇₅ · σ/√n`
**Cas 2** — variance inconnue : `IC = X̄ ± t₀.₉₇₅,df · s/√n`
(où `s` = écart-type échantillonnal, `df = n-1`)""", ),
code("""mu, sigma = 10.0, 3.0
rng = np.random.default_rng(0)
data = rng.normal(mu, sigma, size=60)
xbar = data.mean(); se = sigma / np.sqrt(len(data))
z975 = stats.norm.ppf(0.975)
ic_mu = (xbar - z975*se, xbar + z975*se)
print(f"n={len(data)}, X̄={xbar:.3f}, σ connue={sigma}")
print(f"IC 95% (normale)  : [{ic_mu[0]:.3f}, {ic_mu[1]:.3f}]  contient μ? {ic_mu[0]<=mu<=ic_mu[1]}")

data2 = rng.normal(mu, sigma, size=60)
xbar2 = data2.mean(); s2 = data2.std(ddof=1); se2 = s2/np.sqrt(len(data2))
t975 = stats.t.ppf(0.975, df=len(data2)-1)
ic_t = (xbar2 - t975*se2, xbar2 + t975*se2)
print(f"IC 95% (t, s inconnue={s2:.3f}): [{ic_t[0]:.3f}, {ic_t[1]:.3f}]  contient μ? {ic_t[0]<=mu<=ic_t[1]}")
print(f"  z={z975:.4f} vs t={t975:.4f} (t plus large)")"""),
md("""## 5 · Régression linéaire

### Simple : `y = β₀ + β₁ x + ε`
Les paramètres sont estimés par **moindres carrés** :
```
minimize  Σ (yᵢ - β₀ - β₁ xᵢ)²
```
Solution close-form (rappel T01 — système normal) :
```
β₁ = S_xy / S_xx  =  Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²
β₀ = ȳ - β₁ x̄
R² = 1 - SSE/SST   (fraction de variance expliquée)
```

### Multiple : `y = β₀ + β₁x₁ + ... + βₖxₖ + ε`
Même principe en matriciel : `β̂ = (XᵀX)⁻¹ Xᵀy`.
Si `XᵀX` est mal conditionné → **régularisation** (Ridge, Lasso, ElasticNet)
— on y reviendra au T07.""", ),
code("""rng = np.random.default_rng(42)
x = rng.uniform(0, 10, 100)
y = 1.5 * x + 2 + rng.normal(0, 1, 100)
reg = lm.LinearRegression().fit(x.reshape(-1, 1), y)
pred = reg.predict(x.reshape(-1, 1))
r2 = reg.score(x.reshape(-1, 1), y)
print(f"β₁ estimé = {reg.coef_[0]:.3f} (vrai 1.5),  β₀ = {reg.intercept_:.3f} (vrai 2)")
print(f"R² = {r2:.4f}")
assert abs(reg.coef_[0] - 1.5) < 0.15 and r2 > 0.7

# Régression multiple
Xm = rng.uniform(-3, 3, (150, 2))
beta_true = np.array([2.0, -1.0])
ym = Xm @ beta_true + 0.5 + rng.normal(0, 0.5, 150)
regM = lm.LinearRegression().fit(Xm, ym)
print(f"\\nMultiple : coefs = {regM.coef_} (vrais [2,-1]), "
      f"intercepte={regM.intercept_:.3f} (vrai 0.5)")
assert np.allclose(regM.coef_, beta_true, atol=0.15)"""),
md("""## 6 · Corrélation & coefficient de détermination

**Corrélation de Pearson** `r` :
```
r = cov(x,y) / (sd_x · sd_y) ∈ [-1, 1]
```
Mesure de la **linéarité** de la relation entre deux variables.
`r = 0` : pas de relation **linéaire** (il peut y en avoir une non-linéaire !).
`r²` (coefficient de détermination) : fraction de la variance de `y`
expliquée par `x` (dans un modèle linéaire).

**Pièges classiques :**
- `r` ≠ causalité
- `r` sensible aux outliers
- `r = 0` ne signifie pas « indépendance » (seulement « non linéaire »)""", ),
code("""df = pd.DataFrame({'A': rng.normal(0, 1, 50)})
df['B'] = df['A'] * 0.8 + rng.normal(0, 1, 50)
df['C'] = rng.normal(0, 1, 50)
corr_AB = np.corrcoef(df['A'], df['B'])[0, 1]
corr_AC = np.corrcoef(df['A'], df['C'])[0, 1]
print(f"corr(A,B) = {corr_AB:.3f}  (r² = {corr_AB**2:.3f})")
print(f"corr(A,C) = {corr_AC:.3f}  (≈ 0 : indépendantes)")
assert abs(corr_AB) > 0.5 and abs(corr_AC) < 0.3

# Corrélation ≠ indépendance : U-shape
x_u = rng.normal(0, 1, 100)
y_u = x_u**2 + rng.normal(0, 0.5, 100)
print(f"\\ncorr(x, x²+bruit) = {np.corrcoef(x_u, y_u)[0,1]:.3f}  (≈ 0, pourtant y dépend de x)")"""),
md("""## 🎯 Exercices corrigés
"""),
code("""# Ex 1 : loi de X̄ pour X~N(5,2), n=100
mu1, sig1, n1 = 5, 2, 100
print(f"Ex1 : X̄ ~ N({mu1}, {sig1**2/n1:.4f}) ; sd(X̄) = {sig1/np.sqrt(n1):.4f}")
assert np.isclose(sig1/np.sqrt(n1), 0.2)"""),
code("""# Ex 2 : IC de moyenne (t) sur 10 mesures
m = np.array([52, 49, 50, 55, 51, 48, 53, 50, 49, 52], dtype=float)
mb = m.mean(); ms = m.std(ddof=1); n2 = len(m)
tc = stats.t.ppf(0.975, df=n2-1)
ic2 = (mb - tc*ms/np.sqrt(n2), mb + tc*ms/np.sqrt(n2))
print(f"Ex2 : X̄={mb:.2f}, s={ms:.3f}, t={tc:.3f} -> IC95=[{ic2[0]:.3f},{ic2[1]:.3f}]")"""),
code("""# Ex 3 : P(X̄ > 105) via TCL ; population N(100,15), n=9
mu3, sig3, n3 = 100, 15, 9
se3 = sig3 / np.sqrt(n3)
p3 = 1 - stats.norm.cdf(105, mu3, se3)
print(f"Ex3 : P(X̄>105) = {p3:.4f}  (z = (105-100)/{se3:.1f} = 1)")
assert np.isclose(p3, 0.1587, atol=1e-3)"""),
code("""# Ex 4 : régression par moindres carrés à la main
# Points (0,1),(1,3),(2,4),(3,6)
Xc = np.array([0, 1, 2, 3]); Yc = np.array([1, 3, 4, 6])
xc, yc = Xc.mean(), Yc.mean()
beta = np.sum((Xc - xc) * (Yc - yc)) / np.sum((Xc - xc) ** 2)
alpha = yc - beta * xc
print(f"Ex4 : pente={beta:.3f}, ordonnée={alpha:.3f}")
assert np.isclose(beta, 1.6) and np.isclose(alpha, 1.1)"""),
code("""# Ex 5 : IC pour une proportion
n5, k5 = 400, 280
phat = k5 / n5
z5 = stats.norm.ppf(0.975)
ic5 = (phat - z5*np.sqrt(phat*(1-phat)/n5),
       phat + z5*np.sqrt(phat*(1-phat)/n5))
print(f"Ex5 : p-hat={phat:.3f} -> IC95=[{ic5[0]:.4f},{ic5[1]:.4f}]")

print("\\n✅ T05 terminé.")"""),
]

# ======================================================================
# NOTEBOOK 06 — TESTS & INFÉRENCE
# ======================================================================
cells06 = [
md("""# T06 · Inférence statistique & tests — ADIF72

**Objectifs :**
- Tests paramétriques (z, t, binomial, chi2)
- Tests non paramétriques (Mann-Whitney, Wilcoxon, Kruskal, Shapiro)
- ANOVA (comparaison de 3+ groupes) + post-hoc Tukey
- Chi-deux (indépendance)
- Statistique bayésienne (Bêta-Binomiale)
- Théorie de la décision (coûts)

**Idée maîtresse :** chaque test est une **procédure de décision** :
« rejeter H0 ou non ? » — on contrôle la probabilité d'erreur de type I
(`α`) et on mesurera le pouvoir du test (probabilité de rejeter H0
**quand H1 est vraie**).""", ),
code("""import numpy as np
from scipy import stats
np.set_printoptions(precision=4, suppress=True, linewidth=100)
print("Prêt.")"""),
md("""## 1 · Tests paramétriques

Un test **paramétrique** suppose que les données suivent une loi connue
(normale, binomiale, etc.). Avantages : plus puissant si l'hypothèse
est vraie. Inconvénients : faux négatifs si elle est fausse.

| Test | Question | Statistique |
|---|---|---|
| **z** (1 moyenne) | `μ = μ₀ ?` | z = (X̄-μ₀)/(σ/√n) |
| **t** (1 échantillon) | `μ = μ₀ ?` (σ inconnu) | t = (X̄-μ₀)/(s/√n) |
| **t** (2 échantillons) | `μ₁ = μ₂ ?` | t = (X̄₁-X̄₂)/SE |
| **binomial** | `p = p₀ ?` | P(X≥k) |
| **chi2** (conformité) | distribution = attendue ? | χ² = Σ(obs-att)²/att |
| **chi2** (indépendance) | 2 catégories indépendantes ? | χ² sur tableau |

**Le p-value est une probabilité conditionnelle :**
`p = P(statistique aussi extrême | H0 vraie)`.
Ce n'est PAS `P(H0 vraie | données)`.""", ),
code("""rng = np.random.default_rng(3)
# 1a) z-test (variance connue)
n = 100
sample_z = rng.normal(0.3, 1.0, n)
z = (sample_z.mean() - 0) / (1.0/np.sqrt(n))
p_val = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"1a) z-test : z={z:.3f}, p={p_val:.4f} -> {'rejeter' if p_val<0.05 else 'non'} H0")
assert p_val < 0.05

# 1b) t-test 2 échantillons
g1 = rng.normal(10, 1, 40); g2 = rng.normal(10.8, 1, 40)
t, p_t = stats.ttest_ind(g1, g2, equal_var=True)
print(f"1b) t-test : t={t:.3f}, p={p_t:.4f} -> {'différent' if p_t<0.05 else 'pas de diff'}")
assert p_t < 0.05

# 1c) chi2 indépendance
table = np.array([[20, 30, 45], [30, 25, 35]])
chi2, p_chi, dof, _ = stats.chi2_contingency(table)
print(f"1c) chi2-indépendance : chi2={chi2:.3f}, dof={dof}, p={p_chi:.4f}")

# 1d) binomial
res_b = stats.binomtest(45, 100, 0.5, alternative='two-sided')
print(f"1d) binomial : p={res_b.pvalue:.4f} -> "
      f"{'rejet' if res_b.pvalue<0.05 else 'non'} de p=0.5")"""),
md("""## 2 · Tests non paramétriques

Ils ne font **pas d'hypothèse sur la distribution**. Robustes aux outliers,
valables pour des échelles ordinales. **Moins puissants** que les tests
paramétriques si les hypothèses de ceux-ci sont satisfaites.

| Test paramétrique | Test non paramétrique (équivalent) |
|---|---|
| t-test 1 | **Wilcoxon signé** |
| t-test 2 échantillons | **Mann-Whitney U** |
| ANOVA | **Kruskal-Wallis** |
| (prérequis normalité) | **Shapiro-Wilk** |

**Quand les utiliser ?**
- données non normales (vérifier avec Shapiro)
- petites tailles d'échantillon
- échelles ordininales (questionnaires Likert)""", ),
code("""rng = np.random.default_rng(0)
# 2a) Mann-Whitney
a = rng.normal(5, 1, 50); b = np.abs(rng.normal(5, 1, 50)) + rng.normal(0,0.1,50)
u, p_u = stats.mannwhitneyu(a, b, alternative='two-sided')
print(f"2a) Mann-Whitney : U={u:.1f}, p={p_u:.4f}")

# 2b) Wilcoxon signé
before = rng.normal(10, 1, 30); after = before + rng.normal(0.5, 0.5, 30)
w, p_w = stats.wilcoxon(before, after, alternative='two-sided')
print(f"2b) Wilcoxon signé : W={w:.1f}, p={p_w:.4f}")

# 2c) Kruskal-Wallis
gA = rng.normal(0, 1, 30); gB = rng.normal(0.5, 1, 30); gC = rng.normal(1, 1, 30)
k, p_k = stats.kruskal(gA, gB, gC)
print(f"2c) Kruskal : H={k:.3f}, p={p_k:.4f} -> {'diff' if p_k<0.05 else 'n/d'}")
assert p_k < 0.05

# 2d) Shapiro (prérequis)
norm_s = rng.normal(0, 1, 50); skew_s = rng.exponential(1, 50)
w_sp, p_sp = stats.shapiro(norm_s)
w_sk, p_sk = stats.shapiro(skew_s)
print(f"2d) Shapiro (normale) : p={p_sp:.3f} ; (expo) : p={p_sk:.3f}")"""),
md("""## 3 · ANOVA (Analyse de variance)

Pour comparer **3 ou plus** moyennes. H0 : toutes les moyennes sont égales.
L'idée : décomposer la variance totale en
- **variabilité entre groupes** (traitement)
- **variabilité dans les groupes** (bruit)

Statistique `F = (SSB/df_between) / (SSW/df_within)`.
`F` grand → les différences entre groupes dominent le bruit → rejeter.

**Post-hoc** : si l'ANOVA est significative, il faut savoir **quelles paires**
diffèrent. On utilise **Tukey HSD** (plus strict que des t-tests multiples,
qui gonflerait `α`).""", ),
code("""rng = np.random.default_rng(0)
t1 = rng.normal(20, 2, 25); t2 = rng.normal(20, 2, 25); t3 = rng.normal(24, 2, 25)
f, p_anova = stats.f_oneway(t1, t2, t3)
print(f"ANOVA : F={f:.3f}, p={p_anova:.4f} -> "
      f"{'moyennes diff' if p_anova<0.05 else 'pas de diff'}")
assert p_anova < 0.05

def tukey_pairwise(groups, alpha=0.05):
    gs = list(groups); k = len(gs)
    mse = sum(np.var(g, ddof=1) for g in gs) / k
    df = sum(len(g) - 1 for g in gs)
    q = stats.studentized_range.ppf(1 - alpha, k, df)
    hsd = q * np.sqrt(mse / min(len(g) for g in gs))
    res = {}
    for i in range(k):
        for j in range(i + 1, k):
            res[f"g{i}-g{j}"] = abs(gs[i].mean() - gs[j].mean()) > hsd
    return res, hsd

posthoc, hsd = tukey_pairwise([t1, t2, t3])
print(f"HSD = {hsd:.3f}")
for pair, sig in posthoc.items():
    print(f"  {pair}: {'SIGNIFICATIF' if sig else 'n/d'}")
assert posthoc['g0-g2'] and not posthoc['g0-g1']"""),
md("""## 4 · Test du chi-deux (indépendance)

Pour un **tableau de contingence** : vérifie si 2 variables catégorielles
sont indépendantes.
```
H0 : les variables sont indépendantes
χ² = Σ (obs-att)²/att
df = (nb_lignes - 1)(nb_col - 1)
```
Condition d'usage : effectifs attendus `≥ 5` (sinon : Fisher exact).""", ),
code("""rng = np.random.default_rng(0)
obs = np.array([[45, 30, 25], [35, 40, 25]])
chi2, p, dof, exp = stats.chi2_contingency(obs)
print(f"Chi2={chi2:.3f}, dof={dof}, p={p:.4f}")
print(f"-> {'indépendants' if p>0.05 else 'liés'}")
print("Effectifs attendus :\\n", np.round(exp, 1))"""),
md("""## 5 · Statistique bayésienne (Bêta-Binomiale)

Le **bayésien** est différent du **frequentiste** :
- Frequentiste : le paramètre `p` est **fixé** ; on raisonne sur la
  distribution des données.
- Bayésien : le paramètre `p` est **aléatoire** (a une distribution
  de probabilité) ; on met à jour le **prior** avec les données.

Pour la proportion `p` avec des essais binomiaux :
- **Prior** `p ~ Bêta(α, β)` (uniforme si α=β=1)
- **Vraisemblance** : `k` succès sur `n` essais
- **Posterior** : `p | données ~ Bêta(α+k, β+n-k)` (conjugaison)""", ),
code("""alpha, beta = 1, 1
s, e = 60, 40
a_post, b_post = alpha + s, beta + e
p_gt_05 = 1 - stats.beta.cdf(0.5, a_post, b_post)
print(f"Prior Bêta({alpha},{beta}) + {s}/{s+e} -> Posterior Bêta({a_post},{b_post})")
print(f"E[p] = {a_post/(a_post+b_post):.4f}")
print(f"P(p > 0.5) = {p_gt_05:.5f}")
assert p_gt_05 > 0.97"""),
md("""## 6 · Théorie de la décision (coûts)

Quand les erreurs sont **asymétriques** (faux négatif coûte plus que
faux positif), le critère « p < 0.05 » ne suffit plus. On utilise
la **matrice de coût** :
```
             Décision : Malade    Décision : Sain
Réel Malade :  0 (bon)      100 (FN)
Réel Sain   :  20 (FP)      0 (bon)
```
La décision optimale minimise le **coût attendu** :
`E[cost | décision] = P(réel) × cost`.
C'est le fondement de la décision en médecine, finance, ML (seuils).""", ),
code("""C = np.array([[0, 100], [20, 0]])
p_malade = 0.5
cost_dec_malade = p_malade * C[0, 0] + (1 - p_malade) * C[1, 0]
cost_dec_sain = p_malade * C[0, 1] + (1 - p_malade) * C[1, 1]
print(f"Coût moyen 'malade' = {cost_dec_malade:.1f}")
print(f"Coût moyen 'sain'   = {cost_dec_sain:.1f}")
print(f"-> Choisir {'malade' if cost_dec_malade<cost_dec_sain else 'sain'} "
      f"(minimise le coût)")
assert cost_dec_malade < cost_dec_sain"""),
md("""## 🎯 Exercices corrigés
"""),
code("""# Ex 1 : test t 1 échantillon (H0 : μ=100)
r = np.random.default_rng(11)
data = r.normal(103, 4, 15)
t, p = stats.ttest_1samp(data, 100)
print(f"Ex1 : X̄={data.mean():.2f}, t={t:.3f}, p={p:.4f} -> "
      f"{'≠100' if p<0.05 else '=100'}")
assert p < 0.05"""),
code("""# Ex 2 : test de proportions (z)
phat = 120 / 200
z = (phat - 0.5) / np.sqrt(0.5*0.5/200)
p = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"Ex2 : z={z:.3f}, p={p:.4f} -> {'p≠0.5' if p<0.05 else 'p=0.5'}")
assert p < 0.05"""),
code("""# Ex 3 : ANOVA 1 facteur (grouppes de même moyenne)
r2 = np.random.default_rng(21)
A = r2.normal(0, 1, 20); B = r2.normal(0, 1, 20); C = r2.normal(0, 1, 20)
f, p = stats.f_oneway(A, B, C)
print(f"Ex3 : F={f:.3f}, p={p:.3f} -> {'diff' if p<0.05 else 'n/d'} (même moyenne -> p élevé)")
assert p > 0.05"""),
code("""# Ex 4 : chi2 (conformité)
obs4 = np.array([30, 20, 25, 25])
chi2, p = stats.chisquare(obs4, [25, 25, 25, 25])
print(f"Ex4 : chi2={chi2:.3f}, p={p:.4f} -> {'non conforme' if p<0.05 else 'conforme'}")
assert p > 0.05"""),
code("""# Ex 5 : Mann-Whitney
r3 = np.random.default_rng(31)
x = r3.exponential(1, 40); y = r3.exponential(1.5, 40)
u, p = stats.mannwhitneyu(x, y, alternative='two-sided')
print(f"Ex5 : U={u:.1f}, p={p:.4f} -> "
      f"{'distributions diff' if p<0.05 else 'n/d'}")

print("\\n✅ T06 terminé.")"""),
]

specs = [
    ('04_probabilités.ipynb', cells04, 'ADIF72 - T04 Probabilites'),
    ('05_modelisation_statistique.ipynb', cells05, 'ADIF72 - T05 Modelisation Statistique'),
    ('06_tests_et_inference.ipynb', cells06, 'ADIF72 - T06 Tests Et Inference'),
]
for name, cells, title in specs:
    nb = make_nb(cells, title)
    path = os.path.join(BASE, 'notebooks', name)
    nbf.write(nb, path)
    print("wrote", path, "  cells:", len(cells))
