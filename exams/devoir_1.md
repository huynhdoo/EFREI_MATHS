# DEVOIR 1 — ADIF72 · Mathematics for Data Scientists

**Durée : 2 h — Calculatrice non autorisée — Sans document**
**Barème : 20 points — DE = 60 % de la note finale**

---

## PARTIE A — Algèbre linéaire & géométrie (6 points)

**A1 (1.5 pt)** Soit le système :
```
 2x + y − z = 4
  x + 2y + z = 5
  x − y + 2z = 1
```
Résoudre le système. Montrer que votre solution vérifie les trois équations.

**A2 (1 pt)** Soit `M = [[4, 2], [3, 1]]`.
a) Calculer `det(M)` par la formule `ad − bc`.
b) Donner `M⁻¹` à l'aide de la formule `M⁻¹ = (1/det)·[[d, −b], [−c, a]]`.
c) Vérifier que `M · M⁻¹ = I`.

**A3 (1.5 pt)** Soit `P = [[3, 1], [1, 3]]`.
a) Établir le polynôme caractéristique : `det(P − λI) = 0`.
b) En déduire les valeurs propres (en utilisant le discriminant).
c) Donner les vecteurs propres associés et les normaliser en norme 1.
d) Vérifier que la somme des valeurs propres égale la trace de P.

**A4 (1 pt)** Soit `u = (2, 1, 2)` et `v = (1, −1, 1)`.
a) Calculer `u·v` et les normes `‖u‖`, `‖v‖`.
b) Donner le cosinus de l'angle entre u et v, puis l'angle en degrés.

**A5 (1 pt)** Soit `m = (3, 4)` et `n = (1, 2)`.
a) Calculer le coefficient de projection `α = (m·n)/(n·n)`.
b) En déduire `proj_n(m) = α·n`.
c) Montrer que le résidu `m − proj_n(m)` est orthogonal à n (produit scalaire = 0).

---

## PARTIE B — Calcul, optimisation & probabilités (6 points)

**B1 (1.5 pt)** Soit `f(x) = x³ − 3x`.
a) Calculer `f'(x)` puis `f''(x)`.
b) Trouver les points critiques (`f'(x) = 0`).
c) Classer chacun (minimum / maximum / autre) avec le signe de `f''` et calculer `f` en ces points.

**B2 (1.5 pt)** On veut minimiser `f(x) = (x−3)² + (x+1)²` par descente de gradient,
en partant de `x₀ = 5` avec un pas `α = 0.2`.
a) Calculer le gradient `f'(x)`.
b) Écrire l'itération de descente de gradient `x_{k+1} = x_k − α·f'(x_k)`.
c) Calculer les 3 premiers itérés `x₁, x₂, x₃` (valeurs exactes ou arrondies à 4 décimales).
d) Vers quelle valeur l'algorithme converge-t-il ?

**B3 (1.5 pt)** Un test de dépistage : prévalence `P(M) = 1 %`, sensibilité
`P(+|M) = 0.99`, taux de faux positifs `P(+|S) = 0.05`.
a) Calculer `P(+)` (probabilité d'un résultat positif, en utilisant le théorème de la probabilité totale).
b) Calculer `P(M|+)` (théorème de Bayes) — exprimer le résultat en fraction simplifiée et en décimal.
c) Interpréter ce résultat (paradoxe du taux de base).

**B4 (1.5 pt)** Soit `X ~ N(60, 10²)`.
a) Calculer `P(50 < X < 70)` en standardisant.
b) Calculer `P(X > 80)`.
c) Donner le 95ᵉ percentile de X.
(Rappels : Φ(1) ≈ 0.8413, Φ(2) ≈ 0.9772, z₀.₉₅ ≈ 1.645)

---

## PARTIE C — Statistique, tests & régression (8 points)

**C1 (1 pt)** L'échantillon `[2, 4, 4, 4, 5, 5, 7, 9]` (n = 8) est donné.
a) Calculer la moyenne.
b) Calculer la variance échantillonnale (division par n−1).
c) Donner l'écart-type.

**C2 (1.5 pt)** Pour estimer une moyenne μ (σ inconnu), on a :
`n = 25`, `X̄ = 10`, `s = 2.5`.
a) Indiquer la loi à utiliser et les degrés de liberté.
b) Donner la valeur critique `t₀.₉₇₅,₂₄ ≈ 2.064`.
c) Construire l'IC95 pour μ.

**C3 (2 pt)** Deux groupes de patients, traités par deux médicaments :
- Groupe 1 : n₁ = 30, X̄₁ = 52, s₁ = 5
- Groupe 2 : n₂ = 30, X̄₂ = 55, s₂ = 4

On teste `H₀ : μ₁ = μ₂` vs `H₁ : μ₁ ≠ μ₂` (variances supposées hétérogènes).
a) Calculer l'erreur standard `se = √(s₁²/n₁ + s₂²/n₂)`.
b) Calculer la statistique t.
c) Avec `t₀.₉₇₅,₅₈ ≈ 2.002`, conclure (α = 0.05). Justifier.

**C4 (3.5 pt)** Régression linéaire. On dispose des 4 points `(0,1), (1,3), (2,4), (3,6)`.
a) Calculer `x̄`, `ȳ`, `Sxx = Σ(xᵢ−x̄)²` et `Sxy = Σ(xᵢ−x̄)(yᵢ−ȳ)`.
b) En déduire la pente β et l'ordonnée à l'origine α (moindres carrés).
c) Calculer SSE (somme des carrés des erreurs) et SST (somme des carrés totaux).
d) Calculer R².
e) Faire une prédiction pour x = 4.

---
*Fin du devoir — vérifiez vos réponses avant de rendre.*
