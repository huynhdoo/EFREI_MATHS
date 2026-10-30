# DEVOIR 4 — ADIF72 · Mathematics for Data Scientists

**Durée : 2 h — Calculatrice non autorisée — Sans document**
**Barème : 20 points — DE = 60 % de la note finale**

---

## PARTIE A — Algèbre linéaire & géométrie (6 points)

**A1 (1.5 pt)** Soit `A = [[1, 3], [2, 1]]`.
a) Calculer `det(A)`.
b) Donner `A⁻¹`.
c) Vérifier le coefficient (2,1) de `A⁻¹·A`.

**A2 (1.5 pt)** Soit `Q = [[−1, 2], [2, 1]]`.
a) Établir le polynôme caractéristique : `det(Q − λI)`.
b) Trouver les valeurs propres (discriminant).
c) Donner un vecteur propre pour λ₁. Vérifier `Q·v₁ = λ₁·v₁`.

**A3 (1 pt)** Soit `p = (1, 2, 2)` et `q = (2, 1, 2)`.
a) Calculer `p·q`.
b) Donner `‖p‖` et `‖q‖`.
c) Donner `cos θ` (angle entre p et q).

**A4 (1 pt)** Un vecteur `n = (1, 0, −1)` est normal au plan `Π : x − z = 3`.
a) Montrer que le point `A₀ = (3, 0, 0)` appartient à Π.
b) Écrire l'équation vectorielle du plan : `(X − A₀)·n = 0`.
c) Le point `B = (2, 4, 1)` appartient-il à Π ? Justifier.

**A5 (1.5 pt)** Soit `C = [[1, −1, 2], [0, 2, −1], [1, 0, 1]]`.
a) Calculer `det(C)`.
b) C est-elle inversible ? Si oui, calculer `C⁻¹` (matrice des cofacteurs
   ou méthode de Gauss-Jordan) et vérifier `C·C⁻¹ = I` sur une ligne.

---

## PARTIE B — Calcul, optimisation & probabilités (6 points)

**B1 (1.5 pt)** Soit `X ~ Exp(λ = 1)`.
a) Écrire sa fonction de densité et sa fonction de répartition F(x).
b) Calculer `P(X ≤ 1)`.
c) Calculer la médiane de X (résoudre `F(m) = 0.5`).

**B2 (1.5 pt)** Soit `f(x, y) = x² + y²` et `g(x, y) = (2x, 2y)`.
a) Calculer le gradient `∇f` (dérivées partielles).
b) Montrer que `g` est le gradient de f, et en déduire que le point critique
   de f correspond au nul de g.
c) En un point `(x₀, y₀) = (1, 1)`, donner le vecteur `g(1,1)` et sa direction
   (interprétation : plus grande pente).

**B3 (1.5 pt)** Échantillon : `[2, 4, 4, 4, 5, 5, 7, 9]` (n = 8).
a) Calculer le mode.
b) Calculer la médiane (justifier le choix des rangs).
c) Calculer la moyenne (et la comparer à la médiane).

**B4 (1.5 pt)** Soit `X ~ N(0, 1)` (loi normale centrée réduite).
a) Calculer `P(|X| < 1)` (en utilisant Φ(1) ≈ 0.8413).
b) Calculer `P(X > 1.96)`.
c) Donner le quantile `x₀.₉₇₅` (z₀.₉₇₅) de X.

---

## PARTIE C — Statistique, tests & régression (8 points)

**C1 (2 pt)** Un fabricant contrôle des pièces : H₀ : p = 0.10 (10 % de
défauts) vs H₁ : p ≠ 0.10. Dans un échantillon de n = 100, on observe
8 pièces défectueuses (p̂ = 0.08).
a) Calculer l'erreur standard sous H₀ : `√(p₀(1−p₀)/n)`.
b) Calculer la statistique z.
c) Avec z₀.₉₇₅ = 1.96, conclure à α = 0.05.
d) Donner la p-value (deux queues) approximative.

**C2 (2 pt)** Test du chi-deux d'indépendance. Données :

|  | Produit A | Produit B |
|---|---|---|
| Ville | 40 | 60 |
| Rural | 30 | 70 |

a) Calculer les totaux ligne/colonne et le total global.
b) Calculer les 4 effectifs attendus sous H₀.
c) Calculer χ² = Σ (obs−att)²/att.
d) Avec df = 1 et χ²crit(0.05) = 3.841, conclure (α = 0.05).

**C3 (2 pt)** Comparaison de deux méthodes d'apprentissage (10 étudiants
par méthode) :
- M₁ : n₁ = 10, X̄₁ = 75, s₁ = 5
- M₂ : n₂ = 10, X̄₂ = 78, s₂ = 4

On teste `H₀ : μ₁ = μ₂` (variances hétérogènes supposées).
a) Calculer l'erreur standard `se = √(s₁²/n₁ + s₂²/n₂)`.
b) Calculer la statistique t.
c) Avec df ≈ 17 (formule de Welch) et t₀.₉₇₅,₁₇ ≈ 2.110, conclure
   à α = 0.05.

**C4 (2 pt)** Régression. Données : `(0,0), (1,2), (2,4), (3,6)`.
a) Calculer `x̄`, `ȳ`, `Sxx` et `Sxy`.
b) En déduire β et α.
c) Calculer SSE, SST et R² (interpréter).
d) Prédire y pour x = 4.

---
*Fin du devoir — vérifiez vos réponses avant de rendre.*
