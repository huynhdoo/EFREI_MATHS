# DEVOIR 2 — ADIF72 · Mathematics for Data Scientists

**Durée : 2 h — Calculatrice non autorisée — Sans document**
**Barème : 20 points — DE = 60 % de la note finale**

---

## PARTIE A — Algèbre linéaire & géométrie (6 points)

**A1 (1.5 pt)** Soit `A = [[1, 2, 3], [4, 5, 6]]` et `B = [[1, 0], [0, 1], [1, 1]]`.
a) Déterminer si le produit `A·B` est défini et sa forme (dimensions).
b) Calculer `A·B` élément par élément (montrer au moins 2 calculs de coefficient).

**A2 (1.5 pt)** Soit `M = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]`.
a) Calculer le rang de M (justifier).
b) Montrer que le vecteur `k = (1, −2, 1)` appartient au noyau de M.
c) Que peut-on en déduire sur `det(M)` et l'inversibilité ?

**A3 (1.5 pt)** On veut orthonormaliser la famille `(w₁, w₂)` avec
`w₁ = (1, 1)` et `w₂ = (1, 0)` par Gram-Schmidt.
a) Normaliser `w₁` pour obtenir `q₁`.
b) Calculer la composante de `w₂` projetée sur `q₁`.
c) En déduire `q₂` et vérifier `q₁·q₂ = 0` et `‖q₂‖ = 1`.

**A4 (1 pt)** La matrice `R = [[0, −1], [1, 0]]` représente une rotation.
a) Identifier l'angle de rotation (sens compris).
b) Calculer `R·(3, −4)` et `det(R)`.
c) Montrer que `R·Rᵀ = I`.

**A5 (1 pt)** Soit `C = [[5, 2], [2, 2]]`.
a) Montrer que C est symétrique.
b) Trouver `L` triangulaire inférieure telle que `C = L·Lᵀ` (Cholesky)
   en utilisant `L₁₁ = √5` et les relations `L₂₁L₁₁ = 2`, `L₂₂² = c₂₂ − L₂₁²`.
c) Vérifier le produit.

---

## PARTIE B — Calcul, optimisation & probabilités (6 points)

**B1 (1.5 pt)** Soit `f(x) = e^{x²}`.
a) Appliquer la règle de la chaîne pour trouver `f'(x)`.
b) Calculer `f'(1)` (laisser la réponse avec e).
c) Vérifier par différence centrée (h = 10⁻⁵) que votre résultat est cohérent.

**B2 (1.5 pt)** Minimiser `f(x, y) = x² + y²` sous la contrainte `x + y = 4`.
a) Écrire le lagrangien `L(x, y, λ)`.
b) Donner les conditions de premier ordre (`∂L/∂x = 0`, `∂L/∂y = 0`, `∂L/∂λ = 0`).
c) Résoudre le système et donner le point optimal ainsi que la valeur de λ.
d) Donner la valeur minimale de f.

**B3 (1.5 pt)** Le nombre de pannes machines par jour suit une loi de
Poisson de paramètre λ = 2.
a) Écrire `P(X = k)` pour cette loi.
b) Calculer `P(X = 3)` (en laissant 2! dans le calcul puis le résultat).
c) Calculer `P(X ≥ 2)`.
d) Donner `E[X]` et `Var(X)` pour cette loi.

**B4 (1.5 pt)** Le poids (en kg) d'une population suit `N(100, 15²)`.
On prélève un échantillon de n = 9.
a) Donner la loi de la moyenne d'échantillon X̄ (TCL : moyenne exacte car
   population normale ; préciser μ et σ de X̄).
b) Calculer `P(X̄ > 105)` en standardisant.
(Rappel : Φ(1) ≈ 0.8413)

---

## PARTIE C — Statistique, tests & régression (8 points)

**C1 (1.5 pt)** Soit `X ~ N(5, 2²)` et un échantillon de n = 100.
a) Donner `E[X̄]` et `Var(X̄)`.
b) Calculer `P(4.9 < X̄ < 5.1)`.
(Rappel : Φ(0.5) ≈ 0.6915)

**C2 (1.5 pt)** Dans une enquête de 400 personnes, 280 sont favorables
à une proposition.
a) Calculer `p̂` (estimateur de la proportion).
b) Calculer l'erreur standard de p̂.
c) Construire l'IC95 pour la vraie proportion p
   (z₀.₉₇₅ = 1.96, approximation normale).

**C3 (2 pt)** Trois engrais sont testés sur 4 parcelles chacune.
Hauteurs finales (cm) :
- E₁ = [2, 3, 2, 3]
- E₂ = [4, 5, 4, 4]
- E₃ = [5, 6, 5, 4]

a) Énoncer H₀ et H₁.
b) Calculer les moyennes de groupe et la moyenne globale.
c) Calculer SSB (variation entre groupes) :
   `SSB = Σ nⱼ (mⱼ − mglob)²`.
d) Calculer SSW (variation interne) : somme des Σ(xᵢ − mⱼ)² dans chaque groupe.
e) Calculer F = (SSB/2)/(SSW/9) et conclure avec Fcrit(2,9) = 4.256 (α = 0.05).

**C4 (3.5 pt)** Un fabricant de boulangeries enregistre le nombre de
« oui » (achete) et « non » (n'achète pas) par sexe :

|  | Oui | Non |
|---|---|---|
| Hommes | 30 | 20 |
| Femmes | 20 | 30 |

a) Énoncer H₀ et H₁ (test du chi-deux d'indépendance).
b) Calculer les effectifs totaux ligne/colonne et le total.
c) Calculer les 4 effectifs attendus sous H₀.
d) Calculer χ² = Σ (obs−att)²/att.
e) Avec df = 1 et χ²crit(0.05) = 3.841, conclure (α = 0.05).

---
*Fin du devoir — vérifiez vos réponses avant de rendre.*
