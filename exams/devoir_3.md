# DEVOIR 3 — ADIF72 · Mathematics for Data Scientists

**Durée : 2 h — Calculatrice non autorisée — Sans document**
**Barème : 20 points — DE = 60 % de la note finale**

---

## PARTIE A — Algèbre linéaire & géométrie (6 points)

**A1 (1.5 pt)** Vérifier que `x = (1, 1, 1)` est solution du système :
```
 x + y + z = 3
 2x + y + z = 4
 x + 2y + z = 4
```
Puis montrer que la matrice du système est inversible (déterminant) et
qu'il n'existe donc qu'une seule solution.

**A2 (1.5 pt)** Soit `N = [[5, 2], [7, 4]]`.
a) Calculer `det(N)`.
b) Donner `N⁻¹` (formule 2×2).
c) Vérifier `N⁻¹ · N = I` (au moins le coefficient (1,1)).

**A3 (1.5 pt)** Soit `A = [[0, 1], [1, 0]]`.
a) Établir le polynôme caractéristique et trouver les valeurs propres.
b) Donner un vecteur propre pour chaque valeur propre.
c) Déduire `A²` et `Aⁿ` (n pair / n impair).

**A4 (1 pt)** Soit `a = (3, −4, 0)`.
a) Calculer `‖a‖₁`.
b) Calculer `‖a‖₂`.
c) Calculer `‖a‖∞`.

**A5 (1 pt)** La matrice `P = [[0, 0], [0, 1]]` est la projection sur l'axe y.
a) Montrer que `P² = P` (idempotente).
b) Montrer que `Pᵀ = P` (symétrique).
c) Calculer `P·(5, −2)` et interpréter géométriquement.

---

## PARTIE B — Calcul, optimisation & probabilités (6 points)

**B1 (1.5 pt)** L'algorithme de Newton est `x_{k+1} = x_k − f(x_k)/f'(x_k)`
appliqué à `f(x) = x² − 4` pour trouver √4.
a) Écrire l'itération explicite `x_{k+1} = (x_k + 4/x_k)/2`.
b) Avec `x₀ = 3`, calculer `x₁` (fractions exactes) puis `x₂`.
c) Comment la suite converge-t-elle (qualitativement) ?

**B2 (1.5 pt)** Trouver le minimum de `f(x, y) = x² + y² − x − y`
(sans contrainte).
a) Calculer le gradient.
b) Trouver le point critique.
c) Montrer que le Hessian certifie un minimum global et donner la valeur.

**B3 (1.5 pt)** Une entreprise a 3 usines :
- usine A : 50 % de la production, 5 % de défauts
- usine B : 30 % de la production, 3 % de défauts
- usine C : 20 % de la production, 2 % de défauts
a) Calculer la probabilité qu'un produit soit défectueux (probabilité totale).
b) Un produit est défectueux : quelle est la probabilité qu'il vienne
   de l'usine A (Bayes) ?
c) Calculer la probabilité qu'il ne vienne **pas** de l'usine A.

**B4 (1.5 pt)** Soit `X` exponentielle de taux λ = 2 (densité `2e^{−2x}`, x ≥ 0).
a) Calculer `P(X < 0.5)`.
b) Calculer `E[X]`.
c) Exploiter la propriété mémoire : calculer `P(X > 5 | X > 2)`.

---

## PARTIE C — Statistique, tests & régression (8 points)

**C1 (1.5 pt)** On veut estimer μ, moyenne d'une population de variance σ² = 1,
à partir de n tirages i.i.d. `X₁,...,Xₙ`. Deux estimateurs sont proposés :
`T₁ = X̄` (moyenne de n = 25 tirages) et `T₂ = (X₁ + X₂)/2` (moyenne de 2
tirages seulement).
a) Montrer que les deux estimateurs sont sans biais.
b) Calculer `Var(T₁)` et `Var(T₂)`.
c) Dire lequel est plus efficient (variance plus faible) et pourquoi.

**C2 (1.5 pt)** On sait que pour un estimateur `X̄ ~ N(μ, σ²/n)` avec
`σ = 10`, `n = 25`, `X̄ = 50`.
a) Donner la formule de l'IC95 quand σ est connu.
b) Calculer l'erreur standard.
c) Construire l'IC95 (z₀.₉₇₅ = 1.96).

**C3 (2 pt)** Un sondage : n = 200 personnes, 60 % favorables (p̂ = 0.60).
On teste `H₀ : p = 0.50` vs `H₁ : p ≠ 0.50`.
a) Calculer l'erreur standard sous H₀ : `√(p₀(1−p₀)/n)`.
b) Calculer la statistique z.
c) Avec z₀.₉₇₅ = 1.96, conclure à α = 0.05 et donner la p-value
   (deux queues) approximative.

**C4 (3.5 pt)** Régression. Données : `(1,2), (2,3), (3,5), (4,4), (5,6)`.
a) Calculer `x̄`, `ȳ`, `Sxx = Σ(xᵢ−x̄)²`.
b) Calculer `Sxy = Σ(xᵢ−x̄)(yᵢ−ȳ)`.
c) En déduire β (pente) et α (ordonnée).
d) Calculer SSE, SST et R².
e) Prédire y pour x = 6.

---
*Fin du devoir — vérifiez vos réponses avant de rendre.*
