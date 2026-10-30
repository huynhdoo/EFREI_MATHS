# CORRIGÉ — DEVOIR 4 · ADIF72

## PARTIE A (6 points)

**A1 (1.5)** `det(A) = 1·1 − 3·2 = −5`
`A⁻¹ = (1/−5)·[[1, −3], [−2, 1]] = [[−0.2, 0.6], [0.4, −0.2]]`
Vérif (2,1) : `2·(−0.2) + 1·0.4 = −0.4 + 0.4 = 0` ✓

**A2 (1.5)** `det(Q−λI) = (−1−λ)(1−λ) − 4 = λ² + 0·λ − 5 = 0`
`λ = ±√5` ; `λ₁ = √5 ≈ 2.236`, `λ₂ = −√5 ≈ −2.236`
- λ₁ : `(Q−λ₁I)v = 0` ⇒ `v₁ = (√5−1, 2)` (vérif : Q·v₁ = λ₁·v₁)
- Vérif : `Q·(√5−1, 2)ᵀ = (−(√5−1)+4, 2(√5−1)+2) = (7−√5, 2√5)`
  `= √5·(√5−1, 2)` = `√5·v₁` ✓

**A3 (1)** `p·q = 2 + 2 + 4 = 8` ; `‖p‖ = 3`, `‖q‖ = 3`
`cos θ = 8/9 ≈ 0.8889` ⇒ θ ≈ 27.27°

**A4 (1)**
- a) `3 − 0 = 3` ✓ donc A₀ ∈ Π
- b) `(X − (3,0,0))·(1,0,−1) = 0` ⇒ `(x−3)·1 + (y−0)·0 + (z−0)·(−1) = 0`
  ⇒ `x − z = 3` ✓ (forme normale du plan)
- c) `B = (2,4,1)` : `x − z = 2 − 1 = 1 ≠ 3` ⇒ **B ∉ Π**

**A5 (1.5)** `C = [[1,−1,2],[0,2,−1],[1,0,1]]`
- a) `det(C) = 1·(2·1 − (−1)·0) − (−1)·(0·1 − (−1)·1) + 2·(0·0 − 2·1)`
  `= 2 + 1 − 4 = −1`
- b) `det ≠ 0` ⇒ **inversible**. Cofacteurs :
  - Ligne 1 : Cof(1,1) = +2 ; Cof(1,2) = −1 ; Cof(1,3) = −2
  - Ligne 2 : Cof(2,1) = +1 ; Cof(2,2) = −1 ; Cof(2,3) = −1
  - Ligne 3 : Cof(3,1) = −2 ; Cof(3,2) = +1 ; Cof(3,3) = +2
  Adjointe (transposée des cofacteurs) :
  `Adj = [[2, 1, −2], [−1, −1, 1], [−2, −1, 2]]`
  `C⁻¹ = Adj/det = −Adj = [[−2, −1, 3], [1, 1, −1], [2, 1, −2]]`
  Vérif (ligne 1 × colonnes de C⁻¹) :
  - `(1)(−2)+(−1)(1)+(2)(2) = 1` ✓
  - `(1)(−1)+(−1)(1)+(2)(1) = 0` ✓
  - `(1)(3)+(−1)(−1)+(2)(−2) = 0` ✓

## PARTIE B (6 points)

**B1 (1.5)** `f(x) = e^{−x}` (x ≥ 0) ; `F(x) = 1 − e^{−x}`
- b) `P(X≤1) = F(1) = 1 − e^{−1} ≈ 0.6321`
- c) `F(m) = 0.5` ⇒ `1 − e^{−m} = 0.5` ⇒ `e^{−m} = 0.5` ⇒ `m = ln 2 ≈ 0.6931`

**B2 (1.5)** `∇f = (∂f/∂x, ∂f/∂y) = (2x, 2y)`
- b) `g(x,y) = (2x, 2y) = ∇f(x,y)` — donc `g` **est** le gradient de f
  (chaque composante de g est dérivée partielle de f)
- c) `g(1,1) = (2, 2)` : direction de plus forte augmentation de f en (1,1),
  perpendiculaire à la courbe de niveau.

**B3 (1.5)** Échantillon : 2, 4, 4, 4, 5, 5, 7, 9
- a) **Mode = 4** (fréquence 3, max)
- b) n = 8 pair ⇒ médiane = moyenne des termes de rangs 4 et 5 : `(4 + 5)/2 = 4.5`
- c) Moyenne = (2+4+4+4+5+5+7+9)/8 = 40/8 = **5** ; moyenne > médiane ⇒ distribution légèrement asymétrique (queue à droite).

**B4 (1.5)**
- a) `P(|X|<1) = P(−1<X<1) = Φ(1) − Φ(−1) = 2·0.8413 − 1 = 0.6826`
- b) `P(X>1.96) = 1 − Φ(1.96) ≈ 1 − 0.975 = 0.025`
- c) `x₀.₉₇₅ = z₀.₉₇₅ = 1.96`

## PARTIE C (8 points)

**C1 (2)** `p₀ = 0.10`, `n = 100`, `p̂ = 0.08`
- a) `se₀ = √(0.10·0.90/100) = √0.0009 = 0.03`
- b) `z = (0.08 − 0.10)/0.03 = −2/3 ≈ −0.6667`
- c) `|z| = 0.667 < 1.96` ⇒ **ne pas rejeter H₀** à α = 0.05 : pas de preuve
  statistique que la proportion de défauts diffère de 10 %.
- d) P-value = 2·P(Z > 0.6667) = 2·(1 − Φ(0.6667)) ≈ 2·(1 − 0.7475) ≈ **0.5050**

**C2 (2)** Données 2×2 :
```
         A    B    Total
Ville    40   60   100
Rural    30   70   100
Total    70   130  200
```
- b) Attendus : `E(i,j) = ligne·colonne/total`
  - Ville/A : 100·70/200 = 35 ; Ville/B : 100·130/200 = 65
  - Rural/A : 35 ; Rural/B : 65
- c) `χ² = (40−35)²/35 + (60−65)²/65 + (30−35)²/35 + (70−65)²/65`
  `= 25/35 + 25/65 + 25/35 + 25/65 = 0.7143 + 0.3846 + 0.7143 + 0.3846 = 2.1978`
- d) `χ² = 2.198 < 3.841` ⇒ **ne pas rejeter H₀** à α = 0.05 :
  l'origine (ville/rural) et le produit (A/B) sont statistiquement indépendants.
  (p-value ≈ 0.1382)

**C3 (2)** M₁ : 75 (n=10, s=5) ; M₂ : 78 (n=10, s=4)
- a) `se = √(25/10 + 16/10) = √(4.1) = 2.0248`
- b) `t = (75 − 78)/2.0248 ≈ −1.4816`
- c) `|t| = 1.482 < 2.110 = t₀.₉₇₅,₁₇` ⇒ **ne pas rejeter H₀** à α = 0.05 :
  pas de différence significative entre les deux méthodes dans cet échantillon.
  (p-value ≈ 0.1550)

**C4 (2)** `(0,0),(1,2),(2,4),(3,6)` : points parfaitement alignés sur `y = 2x`
- a) `x̄ = 1.5, ȳ = 3`
  `Sxx = (0−1.5)² + (1−1.5)² + (2−1.5)² + (3−1.5)² = 2.25 + 0.25 + 0.25 + 2.25 = 5`
  `Sxy = (0−1.5)(0−3) + (1−1.5)(2−3) + (2−1.5)(4−3) + (3−1.5)(6−3)`
  `= 4.5 + 0.5 + 0.5 + 4.5 = 10`
- b) `β = 10/5 = 2` ; `α = 3 − 2·1.5 = 0` ⇒ **ŷ = 2x**
- c) Prédits exacts : 0, 2, 4, 6 ⇒ **SSE = 0** ; SST = 20 ; **R² = 1.0**
  (ajustement parfait — aucune variance non expliquée)
- d) `ŷ(4) = 2·4 = 8`
