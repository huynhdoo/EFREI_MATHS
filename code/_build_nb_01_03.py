#!/usr/bin/env python3
# Build notebooks 01, 02, 03 for ADIF72
import nbformat as nbf

NB = nbf.v4

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

# ======================================================================
# NOTEBOOK 01 — ALGÈBRE LINÉAIRE
# ======================================================================
cells01 = [
md("""# T01 · Algèbre linéaire — ADIF72 Mathematics for Data Scientists

**Objectifs du cours :**
- Manipuler les systèmes linéaires, matrices et transformations
- Comprendre rang, bases, sous-espaces (image / noyau)
- Utiliser les valeurs propres et la SVD pour analyser des données

**Pourquoi c'est fondamental en data science :**
| Besoin data science | Outil algébrique |
|---|---|
| Régression linéaire | Résolution de `Xθ = y` / moindres carrés |
| PCA (réduction de dimension) | SVD / valeurs propres de la matrice de covariance |
| Réseaux de neurones | Propagation = chaînes de produits matriciels |
| RecSys (recommandation) | Facteurs matriciels (SVD) |

**Pré-requis :** arithmatique matricielle de base. Si ce n'est pas le cas, relire
Deisenroth ch. 2 ou Lay ch. 1-3.

> 💡 *Conseil étude* : exécuter chaque cellule et **modifier les valeurs** pour
> voir ce qui change. L'algèbre linéaire se comprend en manipulant.
"""),
md("""## 📌 Bloc 0 — Environnement & imports
Installez le stack si besoin (déjà pré-installé sur Colab).
Chaque notebook contient des sorties pré-calculées : vous pouvez les relire
sans rien exécuter, ou « Run All » pour reproduire.""", ),
code("""!pip install -q numpy scipy pandas matplotlib
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)
print("numpy", np.__version__)"""),
md("""## 1 · Opérations matricielles

### Ce qu'on apprend
- **Produit matriciel** `A @ B` : compose deux transformations linéaires.
  `(AB)_{ij} = Σ_k A_{ik} B_{kj}`. C'est **l'opération centrale** de tout le ML.
- **Transpose** `A.T` : échange lignes/colonnes ; apparaît dans les moindres
  carrés (`XᵀX`).
- **Déterminant** `det(A)` : volume ( signé ) transformé ; si `det=0` la matrice
  est **singulière** (non inversible, rang strictement inférieur à n).
- **Inversée** `A⁻¹` : satisfait `AA⁻¹ = I`. N'existe que si `A` est carrée
  ET de rang plein.

### Piège classique
`A @ B ≠ B @ A` en général (le produit matriciel **n'est pas commutatif**).
Vérifiez toujours avec un petit code.""", ),
code("""A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("A @ B =\\n", A @ B)
print("B @ A =\\n", B @ A, " -> différent (non commutatif)")
print("det(A) =", np.linalg.det(A))
print("inv(A) =\\n", np.linalg.inv(A))
print("A @ inv(A) ≈ Identité?\\n", A @ np.linalg.inv(A))"""),
md("""## 2 · Systèmes linéaires  `Ax = b`

### Ce qu'on apprend
Un système de `n` équations à `n` inconnues correspond à une matrice `A`.
- Si `det(A) ≠ 0` : **solution unique** `x = A⁻¹b` (ou mieux : `np.linalg.solve`).
- Si `det(A) = 0` : **0 solution** (incompatible) ou **∞ solutions** (syst. dépendant).

### En data science
La régression par moindres carrés se résout comme un système :
le **système normal** `XᵀX θ = Xᵀy`. Avec une matrice de features mal conditionnée,
`XᵀX` devient singulière → il faut régulariser (Ridge, Lasso) — on y revient au T03.""", ),
code("""# 2x + 3y - z = 5 ;  x - y + 2z = 6 ;  3x + 2y + 2z = 9
A_sys = np.array([[2, 3, -1], [1, -1, 2], [3, 2, 2]])
b_sys = np.array([5, 6, 9])
x = np.linalg.solve(A_sys, b_sys)
print("Solution x =", x)
print("Vérif A@x =", A_sys @ x, "(= b attendu)")

# Cas singulier : 3e ligne = 1re + 2e
A_sing = np.array([[1, 1, 0], [1, -1, 0], [2, 0, 0]])
print("\\nMatrice singulière det = ", np.linalg.det(A_sing),
      " rang =", np.linalg.matrix_rank(A_sing))"""),
md("""## 3 · Rang, bases, sous-espaces

### Concept clé
Toute matrice `A ∈ ℝ^{m×n}` définit une application linéaire `ℝⁿ → ℝᵐ`.
Deux sous-espaces en découlent :
- **Image** (ou *rang*) : tous les vecteurs `A x`. Dimensions = `rang(A)`.
- **Noyau** (*kernel*) : tous les `x` tels que `A x = 0`.

**Théorème fondamental de l'algèbre linéaire** :
```
dim(image A) + dim(noyau A) = n   (nombre de colonnes)
```

### SVD (Singular Value Decomposition)
`A = U Σ Vᵀ` avec `U`, `V` orthogonales, `Σ` diagonale des **singulaires**
(triés décroissants). Les singulaires mesurent l'**amplitude** de A dans
chaque direction.
- Rang(A) = nombre de singulaires non nuls
- Pseudoinverse de A = `V Σ⁻¹ Uᵀ` (résout `Ax=b` au sens moindres carrés)
- Base du noyau = dernières colonnes de `V`""", ),
code("""M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("M de rang", np.linalg.matrix_rank(M), "(2, car col3 = col1 + col2 ?")
print("   Vrai test : col3 - 2*col2 + col1 =", M[:,2] - 2*M[:,1] + M[:,0])

U, S, Vt = np.linalg.svd(M, full_matrices=False)
print("Singulaires de M :", S)
print("Rang effectif (singulaires > tol) :", np.sum(S > 1e-9))

# Base du noyau = dernière colonne de V (ici 1 seule)
null_vec = Vt[-1]
print("Noyau ~ vecteur", null_vec, " -> M @ null_vec =", M @ null_vec)"""),
md("""## 4 · Transformations linéaires (rotation)

Une rotation d'angle `θ` dans le plan :
```
R = [ cosθ  -sinθ ]
    [ sinθ   cosθ ]
```
Propriétés : `det(R)=1`, `R Rᵀ = I` (orthogonale), et la norme est conservée :
`‖Rv‖ = ‖v‖`.
En data science, les rotations apparaissent dans PCA (rotation des axes),
les CNN (invariance rotationnelle approximative), et les embeddings.""", ),
code("""theta = np.pi / 3   # 60°
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
v = np.array([1.0, 0.0])
print("R =\\n", R)
print("Rotation de (1,0) sur 60° ->", R @ v)
print("Norme conservée :", np.isclose(np.linalg.norm(R @ v), np.linalg.norm(v)))
print("R @ R.T ≈ I ?", np.allclose(R @ R.T, np.eye(2)))"""),
md("""## 5 · Valeurs propres

`Av = λv` : le vecteur `v` est simplement **étiré** par le facteur `λ`.
Pour une matrice symétrique : valeurs propres **réelles**, vecteurs
**orthogonaux** → on utilise `np.linalg.eigh` (plus stable).

**Interprétation ML :**
- Matrice de covariance `C = XᵀX/n` → ses valeurs propres = **variances**
  sur les directions principales.
- Les vecteurs propres = les **axes principaux** = PCA.
- En graphes : la valeur propre max du laplacien renseigne sur la connectivité.""", ),
code("""P = np.array([[2, 1], [1, 2]])   # symétrique
eigs, eigvecs = np.linalg.eigh(P)
print("Valeurs propres :", eigs)
print("Vecteurs propres (colonnes) :\\n", eigvecs)
print("Vérif P@v_i = λ_i*v_i :", P @ eigvecs[:, 0],
      "=", eigs[0], "* ", eigvecs[:, 0])
assert np.allclose(P @ eigvecs, eigvecs * eigs)
print("\\nSomme des valeurs propres =", eigs.sum(), "(= trace)")
print("Produit des valeurs propres =", eigs.prod(), "(= det)")"""),
md("""## 6 · SVD — la décomposition universelle

`A = U Σ Vᵀ`. C'est **plus général** que les valeurs propres (travaille
sur toute matrice, pas seulement carrée). Applications phares :
- **Compression** : garder seulement les `k` premiers singulaires →
  approximation `A ≈ Uₖ Σₖ Vₖᵀ` (théorème d'Eckart-Young : meilleure
  approximation de rang `k`).
- **Recommandation** : `X ≈ Uₖ Σₖ Vₖᵀ` sur une matrice utilisateur-article.
- **PCA** : SVD de la matrice centrée = PCA.""", ),
code("""X = np.array([[9, 3, 4],
              [2, 6, 7]])
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print("Singulaires :", S)
print("Reconstruction U@diag(S)@Vt =\\n", U @ np.diag(S) @ Vt)

# Approximation de rang 1 (théorème d'Eckart-Young)
X_r1 = U[:, :1] @ np.diag(S[:1]) @ Vt[:1, :]
print("\\nApproximation rang 1 :\\n", X_r1)
print("Erreur (Frobenius) :||X - X_r1|| =", np.linalg.norm(X - X_r1))"""),
md("""## 7 · Application : PCA manuel par SVD

Données centrées → SVD → les colonnes de `V` (lignes de `Vᵀ`) sont les
composantes principales, et `Sᵢ²` = variance expliquée.

Le ratio `s₁² / Σ sᵢ²` donne la part de variance captée par PC1.
C'est la même chose que sklearn.decomposition.PCA, mais sans l'outil.""", ),
code("""rng = np.random.default_rng(0)
# nuage 2D allongé (corrélation forte entre dimensions)
pts = rng.normal(0, 1, (100, 2)) * np.array([3, 1])
Xc = pts - pts.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
pc1 = Vt[0]
var_expl = (S[0] ** 2) / np.sum(S ** 2)
print("Direction PC1 =", pc1)
print(f"Part de variance expliquée par PC1 = {var_expl:.3f}")

# Vérif avec sklearn
from sklearn.decomposition import PCA
pca = PCA().fit(Xc)
print("sklearn PC1 =", pca.components_[0], "  ratio =", pca.explained_variance_ratio_[0])
assert np.allclose(np.abs(pca.components_[0]), np.abs(pc1))
assert np.isclose(pca.explained_variance_ratio_[0], var_expl)"""),
md("""## 🎯 Exercices corrigés

**Énoncés + corrigés commentés.** Pour s'entraîner : relire sans exécuter,
répondre à la main, puis comparer avec les cellules ci-dessous.""", ),
code("""# Ex 1 : résolvez un système 3x3 (vérifié par substitution)
#   x + 2y + 3z = 6 ;  2x + 3y + 5z = 9 ;  x + 4y + 7z = 10
A1 = np.array([[1, 2, 3], [2, 3, 5], [1, 4, 7]])
b1 = np.array([6, 9, 10])
sol = np.linalg.solve(A1, b1)
print("Ex1 : solution =", sol)
assert np.allclose(A1 @ sol, b1)"""),
code("""# Ex 2 : det, inverse, M²
M2 = np.array([[3, 1], [2, 4]])
print("Ex2 : det(M)=", np.linalg.det(M2),
      " M^-1=\\n", np.linalg.inv(M2),
      " M^2=\\n", M2 @ M2)
assert np.isclose(np.linalg.det(M2), 10)
assert np.allclose(M2 @ np.linalg.inv(M2), np.eye(2))"""),
code("""# Ex 3 : la famille {v1,v2,v3} est-elle une base de R^3 ?
v = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
r = np.linalg.matrix_rank(v)
print("Ex3 : rang =", r, "-> base ?", r == 3)
# 2e vecteur = 2 × 1re vecteur -> linéairement dépendants
assert r < 3"""),
code("""# Ex 4 : valeurs propres (vérif par trace/det)
Q = np.array([[4, 1], [1, 3]])
e, _ = np.linalg.eigh(Q)
print("Ex4 : valeurs propres =", e)
assert np.isclose(sum(e), np.trace(Q))
assert np.isclose(e[0] * e[1], np.linalg.det(Q))"""),
code("""# Ex 5 : projection orthogonale
u = np.array([3.0, 4.0])
for vv in [np.array([1.0, 0.0]), np.array([1.0, 1.0])]:
    proj = (u @ vv) / (vv @ vv) * vv
    print(f"Ex5 : proj de u sur {vv} = {proj}")
assert np.allclose((u @ np.array([1.0, 0.0])) / (np.array([1.0,0.0]) @ np.array([1.0,0.0])) * np.array([1.0,0.0]),
                   [3, 0])

print("\\n✅ T01 terminé. Tous les blocs sont vérifiés.")"""),
]

# ======================================================================
# NOTEBOOK 02 — GÉOMÉTRIE ANALYTIQUE
# ======================================================================
cells02 = [
md("""# T02 · Géométrie analytique — ADIF72

**Objectifs :**
- Normes (L1, L2, L∞, Minkowski) et distances associées
- Produit scalaire, orthogonalité, angles
- Projections et bases orthonormales (Gram-Schmidt)
- Similarité cosinus (applications : recherche, embeddings)

**Pourquoi c'est utile :**
| Besoin data science | Outil géométrique |
|---|---|
| Distance entre points (KNN, clustering) | L2, Manhattan, L∞ |
| Similarité (recherche, RAG, embeddings) | Cosinus, corrélation |
| Orthogonalisation (PCA, QR, SVD) | Gram-Schmidt |
| Reg (Lasso, ElasticNet) | Norme L1 (sparse) |
"""),
code("""import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)
print("Prêt pour la géométrie. numpy", np.__version__)"""),
md("""## 1 · Normes d'un vecteur

Une **norme** est une fonction `‖·‖` qui mesure la « longueur » d'un vecteur :
1. `‖x‖ ≥ 0`, et `‖x‖ = 0 ⇔ x = 0`
2. `‖αx‖ = |α| ‖x‖` (homogénéité)
3. `‖x+y‖ ≤ ‖x‖ + ‖y‖` (inégalité triangulaire)

Les 3 normes usuelles :
| Norme | Formule | Usage typique |
|---|---|---|
| **L1** (Manhattan) | `Σ\\|xᵢ\\|` | Lasso (sparcity) |
| **L2** (Euclidienne) | `√(Σ xᵢ²)` | Moindres carrés |
| **L∞** (max) | `max\\|xᵢ\\|` | Bornes sup |

La **Minkowski** `‖x‖_p = (Σ \\|xᵢ\\|^p)^{1/p}` généralise L1 (p=1), L2 (p=2), L∞ (p→∞).""", ),
code("""x = np.array([3.0, -4.0, 2.0])
print("x =", x)
print("L1      :", np.linalg.norm(x, ord=1))
print("L2      :", np.linalg.norm(x, ord=2))
print("L∞      :", np.linalg.norm(x, ord=np.inf))
print("Mink p=4:", np.linalg.norm(x, ord=4))"""),
md("""## 2 · Produit scalaire, cosinus, angle

Le **produit scalaire** (dot product) de `u, v ∈ ℝⁿ` :
```
u·v = Σ uᵢ vᵢ  =  ‖u‖‖v‖ cos(θ)
```
C'est l'opération la plus importante en géométrie :
- `u·v = 0`  ⇔  **orthogonalité**
- `cos θ > 0`  ⇒  angle aigu (< 90°)
- `cos θ = 0`  ⇒  perpendiculaires
- `cos θ < 0`  ⇒  angle obtus (> 90°)

Le **produit vectoriel** en 3D donne un vecteur perpendiculaire à `u` et `v`
de norme `‖u‖‖v‖ sin θ` (direction = règle de la main droite).""", ),
code("""u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, -1.0, 2.0])
dot = u @ v
cos = dot / (np.linalg.norm(u) * np.linalg.norm(v))
angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
print(f"u·v = {dot}")
print(f"cos(θ) = {cos:.4f}")
print(f"θ = {angle:.2f}°")
# produit vectoriel
print("u × v =", np.cross(u, v))
assert np.isclose((np.cross(u, v)) @ u, 0)  # perpendiculaire à u
assert np.isclose((np.cross(u, v)) @ v, 0)  # et à v"""),
md("""## 3 · Orthogonalité & projection

**Projection orthogonale** de `u` sur `v` :
```
proj_v(u) = (u·v)/(v·v) · v
```
Le résidu `u - proj_v(u)` est **perpendiculaire** à `v`.
C'est la base de la géométrie des moindres carrés : on projette `y`
sur l'image de `X` pour trouver le `θ̂` qui minimise `‖Xθ - y‖`.

**Matrice de projection** sur un sous-espace `W` (spannés par les colonnes
d'une matrice `Q` de colonnes orthogonales) :
```
P_W = Q (Qᵀ Q)⁻¹ Qᵀ
```
Propriétés : `P² = P` (idempotente), `Pᵀ = P` (symétrique), `spec(P) ⊂ {0,1}`.""", ),
code("""def proj(u, v):
    return (u @ v) / (v @ v) * v

u2 = np.array([2.0, 1.0]); v2 = np.array([1.0, 0.0])
p = proj(u2, v2)
print("proj_v(u) =", p)
print("Résidu = u - proj =", u2 - p)
assert np.isclose((u2 - p) @ v2, 0)   # perpendiculaire

# Matrice de projection sur le plan span(e1, e2) dans R3
Q = np.array([[1, 0], [0, 1], [0, 0]])   # plan xy
P = Q @ np.linalg.inv(Q.T @ Q) @ Q.T
print("\\nProjection sur le plan xy (R^3 -> R^3) :\\n", P)
v3 = np.array([1.0, 2.0, 5.0])
print("P @ [1,2,5] =", P @ v3, " (composante z annulée)")
assert np.allclose(P @ P, P)  # idempotente
assert np.allclose(P, P.T)    # symétrique"""),
md("""## 4 · Distances entre points

| Distance | Formule | Sens |
|---|---|---|
| **Euclidienne** | `‖a - b‖₂` | « ligne droite » (physique, géométrie) |
| **Manhattan / L1** | `Σ \\|aᵢ-bᵢ\\|` | « rue de la ville » (grille) |
| **Chebyshev / L∞** | `max \\|aᵢ-bᵢ\\|` | « roque d'échec » (mouvement 3D) |
| **Corrélation** | `1 - r(a,b)` | Indépendante à dérive/échelle (séries) |
| **Cosinus** | `1 - cos` | Indépendante à l'échelle (textes, embeddings) |

En clustering / KNN, le choix de la distance **change le résultat**.
La corrélation et le cosinus sont les plus robustes en data science.""", ),
code("""a = np.array([1.0, 2.0, 3.0]); b = np.array([4.0, 6.0, 3.0])
print(f"Euclidienne  = {np.linalg.norm(a - b):.3f}")
print(f"Manhattan    = {np.abs(a - b).sum():.3f}")
print(f"Chebyshev    = {np.max(np.abs(a - b)):.3f}")

# Distance de corrélation (1 - cosinus des vectres centrés)
def dist_correlation(x, y):
    return 1 - np.corrcoef(x, y)[0, 1]

xs = np.array([1.0, 2, 3, 4, 5])
ys = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
print(f"Corrélation  = {dist_correlation(xs, ys):.5f} (0 = parfaitement corrélées)")"""),
md("""## 5 · Bases orthonormales — Gram-Schmidt

Une **base orthonormale** : des vecteurs de norme 1 mutuellement
perpendiculaires. C'est la « meilleure » base pour :
- diagonaliser une matrice symétrique (théorème spectral)
- décomposer une matrice en QR (`Q` orthogonale, `R` triangulaire)
- le SVD (`U` et `V` sont orthonormales)

**Algorithme de Gram-Schmidt** : orthonormalise une base quelconque
en soustrayant, un à un, la composante projetée sur les précédents.

Variantes :
- **Classique** (ce que nous implémentons) — instable numériquement si les
  vecteurs sont presque parallèles.
- **Modifiée** (MGS) — plus stable, c'est celle qu'utilise `scipy.linalg.qr`.""", ),
code("""def gram_schmidt(V):
    \"\"\"Orthonormalise les colonnes de V.\"\"\"
    Q = []
    for v in V.T:
        q = v.copy()
        for w in Q:
            q = q - (q @ w) * w
        n = np.linalg.norm(q)
        if n < 1e-12:
            continue
        q = q / n
        Q.append(q)
    return np.stack(Q).T

W = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
Qm = gram_schmidt(W)
print("Colonnes d'origine :\\n", W)
print("Base orthonormale Q :\\n", Qm)
print("Q.T @ Q ≈ I ?", np.allclose(Qm.T @ Qm, np.eye(2), atol=1e-8))

# Comparaison avec la QR de numpy
Qq, Rq = np.linalg.qr(W)
print("\\nQR numpy : Q=\\n", Qq, " (signes peuvent différer, mêmes directions)")"""),
md("""## 6 · Application : similarité cosinus (textes, embeddings)

Le **cosinus de similarité** mesure l'angle entre deux vecteurs, **indépendant
de leur longueur**. C'est la métrique reine pour :
- la recherche sémantique
- les embeddings (word2vec, BERT, etc.)
- le clustering de documents
- la détection de plagiat

`cos = 1` : mêmes directions (max similarité) ; `cos = 0` : perpendiculaires ;
`cos = -1` : opposés.""", ),
code("""def cosine_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

# fr/mots simples : TF (term frequency)
doc1 = np.array([3, 1, 2, 0])   # [chat, chien, lapin, oiseau]
doc2 = np.array([2, 1, 3, 0])
doc3 = np.array([0, 0, 5, 4])
print(f"sim(doc1, doc2) = {cosine_sim(doc1, doc2):.3f}")
print(f"sim(doc1, doc3) = {cosine_sim(doc1, doc3):.3f}")
assert cosine_sim(doc1, doc2) > cosine_sim(doc1, doc3)

# Invariance à l'échelle : multiplier doc1 par 100 ne change pas le cosinus
print(f"sim(100*doc1, doc2) = {cosine_sim(100*doc1, doc2):.3f} (invariant !)")"""),
md("""## 🎯 Exercices corrigés
"""),
code("""# Ex 1 : normes de a=(5,-1,2)
a1 = np.array([5.0, -1.0, 2.0])
print(f"Ex1 : L1={np.linalg.norm(a1,1)}, L2={np.linalg.norm(a1,2):.3f}, Linf={np.linalg.norm(a1,np.inf)}")
assert np.linalg.norm(a1, 1) == 8"""),
code("""# Ex 2 : angle entre p=(1,0,0) et q=(0,1,0)
p = np.array([1.0, 0, 0]); q = np.array([0.0, 1, 0])
ang = np.degrees(np.arccos(p @ q / (np.linalg.norm(p) * np.linalg.norm(q))))
print(f"Ex2 : angle(p,q) = {ang}° (orthogonaux)")
assert ang == 90"""),
code("""# Ex 3 : projection de m=(3,4) sur n=(1,2)
# coef = m·n / n·n = (3+8)/(1+4) = 11/5 = 2.2  ->  proj = 2.2*n = (2.2, 4.4)
m = np.array([3.0, 4.0]); n = np.array([1.0, 2.0])
coef = (m @ n) / (n @ n)
pr = coef * n
print(f"Ex3 : coef = {coef}, proj = {pr}")
assert np.allclose(pr, [2.2, 4.4])
assert np.isclose((m - pr) @ n, 0)   # résidu perpendiculaire"""),
code("""# Ex 4 : distance euclidienne entre (0,0,0) et (1,1,1)
d = np.linalg.norm(np.array([1.0, 1, 1]))
print(f"Ex4 : dist = {d:.4f} = √3")
assert np.isclose(d, np.sqrt(3))"""),
code("""# Ex 5 : Gram-Schmidt sur [[1,1],[1,0]]
Vs = np.array([[1.0, 1.0], [1.0, 0.0]])
Qs = gram_schmidt(Vs)
print("Ex5 : Q =\\n", Qs)
print("Q.T @ Q =\\n", Qs.T @ Qs)
assert np.allclose(Qs.T @ Qs, np.eye(2), atol=1e-8)

print("\\n✅ T02 terminé.")"""),
]

# ======================================================================
# NOTEBOOK 03 — CALCUL DIFFÉRENTIEL & OPTIMISATION
# ======================================================================
cells03 = [
md("""# T03 · Calcul différentiel & optimisation — ADIF72

**Objectifs :**
- Dérivées, règle de la chaîne, gradients
- Descente de gradient (à la main, puis ML)
- Convexité (Hessian, valeurs propres)
- Régression linéaire et logistique par descente de gradient

**Pourquoi c'est le cœur du ML :**
Tous les modèles d'apprentissage profond et les SVM, réseaux de neurones,
ensembles (forets), etc. s'entraînent par **descente de gradient** (ou ses
variantes : momentum, Adam, L-BFGS…). Comprendre le calcul du gradient est
indispensable.""", ),
code("""import numpy as np
from scipy import optimize
np.set_printoptions(precision=4, suppress=True, linewidth=100)
print("Prêt. numpy", np.__version__, " scipy", optimize.__name__[:10])"""),
md("""## 1 · Dérivées : analytique vs numérique

### Dérivée analytique (règles)
```
f(x) = x³   ->  f'(x) = 3x²
(f·g)' = f'g + fg'
(f∘g)'(x) = f'(g(x)) · g'(x)    (règle de la chaîne)
```

### Dérivée numérique (différence centrale)
```
f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
```
Précision **O(h²)** — bien meilleure que la différence avant `(f(x+h)-f(x))/h`
(précision O(h)). On prend habituellement `h ≈ 1e-5`.

**Pourquoi ça compte :** en deep learning, on n'a pas toujours la forme
analytique du gradient ; on peut vérifier par différences (c'est le
« *gradient check* »).""", ),
code("""def f(x):
    return x ** 3
def df(x):
    return 3 * x ** 2

def derive_numerique(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

for xv in [0, 1, 2, 3]:
    print(f"x={xv}:  f' analytique={df(xv):.4f}   numérique={derive_numerique(f, xv):.4f}")
assert np.allclose([df(xv) for xv in [0, 1, 2]],
                   [derive_numerique(f, xv) for xv in [0, 1, 2]])

# Règle de la chaîne : g(x) = sin(x^2)  ->  g'(x) = cos(x^2)*2x
x = 0.7
g_prime = np.cos(x ** 2) * 2 * x
print(f"\\nChaîne : g'(0.7) = {g_prime:.4f}")
print(f"Vérif numé  : {derive_numerique(lambda t: np.sin(t**2), x):.4f}")
assert np.isclose(g_prime, derive_numerique(lambda t: np.sin(t ** 2), x))"""),
md("""## 2 · Gradient

Pour `f : ℝⁿ → ℝ`, le **gradient** est le vecteur des dérivées partielles :
```
∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)ᵀ
```
Il pointe dans la direction de la **plus forte croissance** de `f`.
Pour **minimiser**, on va dans la direction opposée : `-∇f`.""", ),
code("""# f(x,y) = x² + 3xy + y²
# df/dx = 2x + 3y ;  df/dy = 3x + 2y
def grad_f(x, y):
    return np.array([2 * x + 3 * y, 3 * x + 2 * y])

g = grad_f(1.0, 2.0)
print("∇f(1,2) =", g)

# Vérif par différences
h = 1e-5
num = np.array([
    ((1+h)**2 + 3*(1+h)*2 + 4 - ((1-h)**2 + 3*(1-h)*2 + 4)) / (2*h),
    (1 + 3*1*(2+h) + (2+h)**2 - (1 + 3*1*(2-h) + (2-h)**2)) / (2*h)])
print("numé      =", num)
assert np.allclose(g, num, atol=1e-3)"""),
md("""## 3 · Descente de gradient (à la main)

Algorithme :
```
x_{t+1} = x_t - α ∇f(x_t)
```
- `α` : **pas de learning rate**. Trop petit = convergence lente.
  Trop grand = divergence / oscillation.
- On choisit souvent `α` par « *trial and error* » ou par **rule of thumb** :
  `α < 2 / L` où `L` est le constant de Lipschitz du gradient.

Exemple 1D : `f(x) = (x-3)² + (x+1)² = 2x² - 4x + 10`. Minimum à `x* = 1`,
`f(x*) = 8`.""", ),
code("""def f1(x):
    return (x - 3) ** 2 + (x + 1) ** 2
def df1(x):
    return 4 * x - 4   # dérivée analytique

def gradient_descent_1d(f, df, x0, lr=0.1, n_iter=200, tol=1e-8):
    x = float(x0)
    hist = [x]
    for i in range(n_iter):
        g = df(x)
        x = x - lr * g
        hist.append(x)
        if abs(g) < tol:
            print(f"  convergé en {i+1} itérations")
            break
    return np.array(hist)

hist = gradient_descent_1d(f1, df1, x0=-5.0, lr=0.1)
print(f"x0 = -5.0, f(x0) = {f1(-5):.2f}")
print(f"x* ≈ {hist[-1]:.4f}, f(x*) ≈ {f1(hist[-1]):.4f}   (optimum : 1, 8)")
assert np.isclose(hist[-1], 1.0, atol=1e-3)"""),
md("""## 4 · Convexité & Hessian

### Test de convexité (matrice symétrique)
Une fonction différentiable à deux fois est **convexe** si son
**Hessian** `H` (matrice des 2ndes dérivées) est semi-défini positif
(les valeurs propres de `H` sont `≥ 0`).
Si les valeurs propres sont **strictement** `> 0`, on est en
**strictement convexe** : le minimum est **unique**.

### Pourquoi ça compte
- Descente de gradient converge globalement pour des fonctions convexes
- Les problèmes de régression linéaire (moindres carrés) sont convexes
- Les réseaux de neurones **ne le sont pas** — d'où les minima locaux
  et le plateau de gradient""", ),
code("""# g(x,y) = x² + y² - 2x - 4y + 10  (convexe : Hessian = 2I > 0)
H = np.array([[2, 0], [0, 2]])
print("Hessian :\\n", H, "  valeurs propres =", np.linalg.eigvalsh(H))

res = optimize.minimize(lambda v: v[0]**2 + v[1]**2 - 2*v[0] - 4*v[1] + 10,
                        x0=[0., 0.], jac=lambda v: [2*v[0]-2, 2*v[1]-4])
print("Minimum scipy :", res.x, "  valeur :", res.fun)
assert np.allclose(res.x, [1, 2]) and np.isclose(res.fun, 5)"""),
md("""## 5 · Régression linéaire par descente de gradient

Modèle : `ŷ = θ x + b`. Perda : MSE `L = (1/n) Σ (ŷᵢ - yᵢ)²`.
Gradients :
```
∂L/∂θ = (2/n) Σ xᵢ (θxᵢ + b - yᵢ)
∂L/∂b = (2/n) Σ (θxᵢ + b - yᵢ)
```
C'est la même chose que les moindres carrés, mais **itératif** (indispensable
si les données ne tiennent pas en mémoire, ou si la matrice est énorme).""", ),
code("""rng = np.random.default_rng(42)
n = 200
X = rng.uniform(-5, 5, n)
theta_true, b_true = 2.5, 1.0
y = theta_true * X + b_true + rng.normal(0, 0.5, n)

def loss_mse(theta, b, X, y):
    pred = theta * X + b
    return np.mean((pred - y) ** 2)

def grad_mse(theta, b, X, y):
    pred = theta * X + b
    err = pred - y
    return (2/len(y)) * np.sum(X * err), (2/len(y)) * np.sum(err)

theta, b = 0.0, 0.0
lr = 0.01
for it in range(2000):
    gt, gb = grad_mse(theta, b, X, y)
    theta -= lr * gt
    b -= lr * gb
print(f"θ estimé = {theta:.3f} (vrai={theta_true}),  b estimé = {b:.3f} (vrai={b_true})")
print(f"Loss finale = {loss_mse(theta, b, X, y):.4f}")
assert abs(theta - theta_true) < 0.1 and abs(b - b_true) < 0.1"""),
md("""## 6 · Régression logistique (classification)

Modèle : `ŷ = σ(xᵀw + b)` avec `σ(z) = 1/(1 + e^{-z})`.
Perda (cross-entropy) :
```
L = -(1/n) Σ [ yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ) ]
```
Gradients :
```
∂L/∂w = (1/n) Xᵀ (ŷ - y)     (le même résidu que la régression linéaire !)
∂L/∂b = (1/n) Σ (ŷ - y)
```
C'est ce que fait sklearn.linear_model.LogisticRegression. Implémenter à la main
→ comprendre comment le modèle apprend le **frontière de décision**.""", ),
code("""X2 = rng.normal(0, 1, (n, 2))
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
    return (1/len(y)) * (X.T @ err), (1/len(y)) * np.sum(err)

w, b = np.zeros(2), 0.0
for it in range(2000):
    gw, gb = log_loss_grad(w, b, X2, y2)
    w -= 0.1 * gw
    b -= 0.1 * gb
acc = np.mean((sigmoid(X2 @ w + b) >= 0.5).astype(int) == y2)
print(f"Vrais poids = {w_true},  estimés = {w}")
print(f"Accuracy = {acc:.3f}   (perdita = {log_loss(w, b, X2, y2):.4f})")
assert acc > 0.7

# Comparaison avec sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr_model = LogisticRegression(max_iter=1000).fit(X2, y2)
print(f"sklearn : weights={lr_model.coef_[0]}, accuracy="
      f"{accuracy_score(y2, lr_model.predict(X2)):.3f}")"""),
md("""## 7 · Récapitulatif — choisir l'optimiseur

| Fonction | Caractéristique | Méthode |
|---|---|---|
| Polynôme de bas degré | convexe, gradient lisse | GD, Newton |
| Réseau de neurones | non-convexe, gradient stochastique | Adam, SGD+momentum |
| SVM | convexe, gros hessian | SMO, L-BFGS |
| Linéaire (moindres carrés) | convexes, gradient exact | GD, closed form |
| Logistique | convexes | Newton / IRLS / GD |
"""),
md("""## 🎯 Exercices corrigés
"""),
code("""# Ex 1 : dérive de f(x) = e^{-x²} (cloche de Gauss)
# f'(x) = -2x e^{-x²}
x0 = 0.5
f1 = lambda x: np.exp(-x**2)
df1 = lambda x: -2 * x * np.exp(-x**2)
print(f"Ex1 : f'({x0}) = {df1(x0):.4f} (analytique)")
print(f"      = {derive_numerique(f1, x0):.4f} (numérique)")
assert np.isclose(df1(x0), derive_numerique(f1, x0))

# Ex 2 : minimiser g(x) = x^4 - 2x² (deux minima, un maximum)
g2 = lambda x: x**4 - 2*x**2
dg2 = lambda x: 4*x**3 - 4*x
crits = np.roots([4, 0, -4, 0])   # racines de g'
print(f"Ex2 : points critiques = {crits}")
for c in crits:
    print(f"   g({c:.2f}) = {g2(c):.3f}, g''({c:.2f}) = {4*c**2-4:.3f}")
assert len(crits) == 3
"""),
code("""print("\\n✅ T03 terminé. Tous les blocs vérifiés.")"""),
]

import os
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'notebooks')
os.makedirs(outdir, exist_ok=True)

specs = [
    ('notebooks/01_algebre_lineaire.ipynb', cells01, 'ADIF72 - T01 Algebre Lineaire'),
    ('notebooks/02_geometrie_analytique.ipynb', cells02, 'ADIF72 - T02 Geometrie Analytique'),
    ('notebooks/03_calcul_optimisation.ipynb', cells03, 'ADIF72 - T03 Calcul Optimisation'),
]
for rel, cells, name in specs:
    nb = make_nb(cells, name)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
    nbf.write(nb, path)
    print("wrote", path, "  cells:", len(cells))
