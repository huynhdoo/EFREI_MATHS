# CORRIGÉ — DEVOIR 2 · ADIF72

## PARTIE A (6 points)

**A1 (1.5)** A est 2×3, B est 3×2 : produit défini, résultat **2×2**.
- (A·B)₁₁ = 1·1 + 2·0 + 3·1 = **4**
- (A·B)₁₂ = 1·0 + 2·1 + 3·1 = **5**
- (A·B)₂₁ = 4·1 + 5·0 + 6·1 = **10**
- (A·B)₂₂ = 4·0 + 5·1 + 6·1 = **11**

`A·B = [[4, 5], [10, 11]]`

**A2 (1.5)** La colonne 2 vaut 2 fois la colonne 1 (`2,4,6 = 2·(1,2,3)`)
⇒ colonnes dépendantes ⇒ **rang(M) = 2** (les colonnes 1 et 3 sont indépendantes).
- `M·k = [1−4+3, 2−8+6, 1−2+1] = [0, 0, 0]` ✓ donc k ∈ noyau.
- `det(M) = 0` ; M **n'est pas inversible**.
- (Théorème fondamental : dim(image) + dim(noyau) = 3 : ici dim(image)=2, dim(noyau)=1)

**A3 (1.5)**
- a) `q₁ = w₁/‖w₁‖ = (1,1)/√2 = (1/√2, 1/√2)`
- b) `proj_{q₁}(w₂) = (w₂·q₁)·q₁ = (1/√2)·(1/√2, 1/√2) = (1/2, 1/2)`
- c) `q₂ = (w₂ − proj)/‖w₂ − proj‖ = (1/2, −1/2)/ (1/√2) = (1/√2, −1/√2)`
  Vérif : `q₁·q₂ = 1/2 − 1/2 = 0` ✓ ; `‖q₂‖ = √(1/2+1/2) = 1` ✓

**A4 (1)**
- a) `R = [[0,−1],[1,0]]` : rotation de **+90°** (sens trigonométrique)
- b) `R·(3,−4) = (0·3 + (−1)(−4), 1·3 + 0·(−4)) = (4, 3)` ; `det(R) = 0+1 = 1`
- c) `R·Rᵀ = [[0,−1],[1,0]]·[[0,1],[−1,0]] = [[1,0],[0,1]] = I` ✓

**A5 (1)** C est symétrique car `c₁₂ = c₂₁ = 2` et `det(C) = 10−4 = 6 > 0`
avec `c₁₁ > 0` ⇒ SPD.
- `L₁₁ = √5`
- `L₂₁ = 2/√5`
- `L₂₂ = √(2 − 4/5) = √(6/5) = √30/5`
`L = [[√5, 0], [2/√5, √30/5]]`
Vérif : `L·Lᵀ = [[5, 2], [2, 4/5+6/5]] = [[5,2],[2,2]]` ✓

## PARTIE B (6 points)

**B1 (1.5)** `f(x) = e^{u(x)}` avec `u = x²` : `f'(x) = u'(x)·e^{u(x)} = 2x·e^{x²}`
- `f'(1) = 2e ≈ 5.4366`
- Différence centrée : `(e^{(1+h)²} − e^{(1−h)²})/(2h) ≈ 5.4366` ✓

**B2 (1.5)** `L = x² + y² − λ(x + y − 4)`
- `∂L/∂x = 2x − λ = 0` ; `∂L/∂y = 2y − λ = 0` ; `∂L/∂λ = −(x+y−4) = 0`
- Des 1res équations : `x = y = λ/2`. Contrainte : `λ/2 + λ/2 = 4` ⇒ `λ = 4`
- **Point optimal : (2, 2)** ; valeur minimale `f(2,2) = 4 + 4 = 8`
- (Géométriquement : le point du droit x+y=4 le plus proche de l'origine)

**B3 (1.5)** `P(X=k) = e^{−λ} λᵏ/k!` avec λ = 2
- `P(X=3) = e^{−2}·2³/3! = 8e^{−2}/6 = 4e^{−2}/3 ≈ 0.1804`
- `P(X≥2) = 1 − P(0) − P(1) = 1 − e^{−2} − 2e^{−2} = 1 − 3e^{−2} ≈ 0.5940`
- `E[X] = Var(X) = λ = 2`

**B4 (1.5)** `X̄ ~ N(100, (15/√9)²) = N(100, 5²)`
`P(X̄ > 105) = 1 − Φ((105−100)/5) = 1 − Φ(1) ≈ 1 − 0.8413 = 0.1587`

## PARTIE C (8 points)

**C1 (1.5)** `E[X̄] = 5` ; `Var(X̄) = 4/100 = 0.04` ⇒ `sd(X̄) = 0.2`
`P(4.9<X̄<5.1) = Φ((5.1−5)/0.2) − Φ((4.9−5)/0.2) = Φ(0.5) − Φ(−0.5)`
`= 2·0.6915 − 1 ≈ 0.3830`

**C2 (1.5)** `p̂ = 280/400 = 0.70`
`se(p̂) = √(p̂(1−p̂)/n) = √(0.7·0.3/400) = √(0.21/400) = √0.000525 ≈ 0.02291`
`IC95 = 0.70 ± 1.96·0.02291 = 0.70 ± 0.0449`
**IC95 ≈ [0.6551, 0.7449]**

**C3 (2)**
- H₀ : μ₁ = μ₂ = μ₃ ; H₁ : au moins un μⱼ différent
- Moyennes : `m₁ = 2.5, m₂ = 4.25, m₃ = 5.0` ; moyenne globale `= 47/12 ≈ 3.9167`
- `SSB = 4·[(2.5−3.9167)² + (4.25−3.9167)² + (5−3.9167)²]`
  `= 4·[2.0069 + 0.1111 + 1.1736] = 4·3.2917 ≈ 13.167`
- SSW interne : E₁ : (−0.5)²+(0.5)²+(−0.5)²+(0.5)² = 1 ; E₂ : 0.75 ; E₃ : 2
  `SSW = 1 + 0.75 + 2 = 3.75`
- `F = (SSB/2)/(SSW/9) = 6.583/0.4167 ≈ 15.80`
- `F = 15.80 > Fcrit = 4.256` ⇒ **rejeter H₀** : les engrais donnent des
  hauteurs significativement différentes (p ≈ 0.0011).

**C4 (3.5)** H₀ : le sexe est indépendant de l'achat ; H₁ : lié
- Totaux : Hommes = 50, Femmes = 50, Oui = 50, Non = 50, Total = 100
- Attendus : `E = ligne·colonne/total = 50·50/100 = 25` pour chaque cellule
- `χ² = (30−25)²/25 + (20−25)²/25 + (20−25)²/25 + (30−25)²/25 = 4·1 = 4`
- `χ² = 4 > 3.841` ⇒ **rejeter H₀** à α = 0.05 (p ≈ 0.0455) :
  l'achat est faiblement lié au sexe dans cet échantillon.
  (Effet marginal : 30/50 = 60 % chez les hommes, 40 % chez les femmes)
