# Plan détaillé des notebooks T01–T07

> ADIF72 — Mathematics for Data Scientists (EFREI, semestre 7, 45 h).
>
> Chaque notebook **T01–T07** est organisé en **6 points-sujets** (`## 1 ·` … `## 6 ·`),
> chacun contenant son **exercice corrigé Pn** (bloc de code, valeurs vérifiées et assertées).
> Le **point 7** (`## 7 ·`) n'est pas un nouveau sujet : il est réservé à la **synthèse
> globale** — tableau récapitulatif des 6 points + exemple intégrateur.
> Les points signalés `*(MML ch. X.Y)*` comblent les gaps entre le syllabus ADIF72 et le
> livre de référence (Deisenroth, Faisal & Ong — *Introduction to Machine Learning*, mml-book.com).

**Convention de lecture**

| Symbole | Signification |
|---|---|
| `Pn` | Point n du notebook (1–6) : théorie + exemples exécutés (sorties visibles) |
| `P7` | Synthèse globale : récap des 6 points + exemple intégrateur (jamais un nouveau sujet) |
| `→ exo Pn` | Exercice corrigé du point n (à refaire sans regarder) |
| `*(MML ch. X.Y)*` | Ajout aligné sur le chapitre X.Y du livre MML |
| `###` | Sous-point (développement du point parent) |

## Sommaire

- **T01 — Algèbre linéaire** (`notebooks/01_algebre_lineaire.ipynb`) — **P1** Opérations matricielles, systèmes & transformations linéaires · **P2** Rang, bases, sous-espaces · **P3** Valeurs propres · **P4** SVD — la décomposition universelle · **P5** PCA, eigendecomposition & Cholesky · **P6** PCA — perspective variable latente · **P7** Synthèse globale + exemple intégrateur (PCA par SVD vs valeurs propres)
- **T02 — Géométrie analytique** (`notebooks/02_geometrie_analytique.ipynb`) — **P1** Normes d'un vecteur · **P2** Produit scalaire, cosinus, angle · **P3** Orthogonalité & projection · **P4** Distances entre points · **P5** Bases orthonormales — Gram-Schmidt · **P6** Application : similarité cosinus (textes, embeddings) · **P7** Synthèse globale + exemple intégrateur (recherche documentaire)
- **T03 — Calcul différentiel & optimisation** (`notebooks/03_calcul_optimisation.ipynb`) — **P1** Dérivées & gradient (analytique et numérique) · **P2** Descente de gradient, convexité & Hessian · **P3** Régression linéaire par descente de gradient · **P4** Logistique, Jacobien & backpropagation · **P5** Hessian, Taylor & Newton · **P6** Optimisation contrainte — Lagrange/KKT · **P7** Synthèse globale (tableau des optimiseurs) + exemple intégrateur (GD vs Newton)
- **T04 — Théorie des probabilités** (`notebooks/04_probabilités.ipynb`) — **P1** Espace probabiliste & axiomes de Kolmogorov · **P2** Variables aléatoires (discrètes et continues) · **P3** Lois usuelles en science des données · **P4** Bayes & test d'hypothèse · **P5** Monte-Carlo & familles exponentielles · **P6** Changement de variable & échantillonnage inverse · **P7** Synthèse globale + exemple intégrateur (Monte-Carlo exponentielle, Bayes)
- **T05 — Modélisation statistique** (`notebooks/05_modelisation_statistique.ipynb`) — **P1** Distribution d'échantillonnage · **P2** Théorème de la Limite Centrale (TCL) · **P3** Estimation — biais & variance d'un estimateur · **P4** Intervalles de confiance (IC) · **P5** Régression linéaire, corrélation & R² · **P6** Régression linéaire bayésienne · **P7** Synthèse globale + exemple intégrateur (fréquentiste / corrélation / bayésien)
- **T06 — Inférence statistique & tests** (`notebooks/06_tests_et_inference.ipynb`) — **P1** Tests paramétriques · **P2** Tests non paramétriques · **P3** ANOVA (Analyse de variance) · **P4** Test du chi-deux (indépendance) · **P5** Statistique bayésienne (Bêta-Binomiale) · **P6** Théorie de la décision (coûts) · **P7** Synthèse globale + exemple intégrateur (A/B test sous trois lectures)
- **T07 — Applications en science des données** (`notebooks/07_applications_datascience.ipynb`) — **P1** Pipeline de modélisation régressive (end-to-end) · **P2** Validation k-fold & biais-variance · **P3** Sur-apprentissage explicite (biais-variance) · **P4** Classification : métriques · **P5** Biais & considérations éthiques · **P6** Tableau récapitulatif — choisir la bonne méthode · **P7** Synthèse globale + exemple intégrateur (pipeline déséquilibré, F1/AUC)

---

## T01 — Algèbre linéaire

**Fichier** : `notebooks/01_algebre_lineaire.ipynb` · **Script miroir** : `code/01_algebre_lineaire.py` · **6 points-sujets + synthèse globale / 7 exercices**

| # | Point | Sous-points | Exercice corrigé (Pn) |
|---|---|---|---|
| 1 | Opérations matricielles, systèmes & transformations linéaires | Systèmes linéaires  `Ax = b`<br>Transformations linéaires (rotation) | Exercice P1 : det/inverse de M=[[3,1],[2,4]] ; rotation 90 deg |
| 2 | Rang, bases, sous-espaces | — | Exercice P2 : la famille est-elle une base de R^3 ? |
| 3 | Valeurs propres | — | Exercice P3 : valeurs propres de [[4,1],[1,3]] |
| 4 | SVD — la décomposition universelle | — | Exercice P4 : erreur de l'approximation SVD de rang 1 |
| 5 | PCA, eigendecomposition & Cholesky `*(MML ch. 4.3-4.4)*` | Eigendecomposition & diagonalisation *(MML ch. 4.4)*<br>Décomposition de Cholesky *(MML ch. 4.3)* | Exercice P5 : Cholesky de [[5,1],[1,2]] + valeurs propres de [[2,1],[1,2]] |
| 6 | PCA — perspective variable latente `*(MML ch. 10.7)*` | — | Exercice P6 : eigendecomposition de corr = [[1,0.9],[0.9,1]] |
| 7 | **Synthèse globale** (récap des 6 points) + exemple intégrateur : PCA par SVD vs valeurs propres, reconstruction rang 1, Cholesky | — | Exercice P7 : projection orthogonale de u=(3,4) sur (1,0) et (1,1) |

---

## T02 — Géométrie analytique

**Fichier** : `notebooks/02_geometrie_analytique.ipynb` · **Script miroir** : `code/02_geometrie_analytique.py` · **6 points-sujets + synthèse globale / 7 exercices**

| # | Point | Sous-points | Exercice corrigé (Pn) |
|---|---|---|---|
| 1 | Normes d'un vecteur | — | Exercice P1 : normes de a=(5,-1,2) |
| 2 | Produit scalaire, cosinus, angle | — | Exercice P2 : angle entre p=(1,0,0) et q=(0,1,0) |
| 3 | Orthogonalité & projection | — | Exercice P3 : projection de m=(3,4) sur n=(1,2) |
| 4 | Distances entre points | — | Exercice P4 : distance + cosinus + projection combines |
| 5 | Bases orthonormales — Gram-Schmidt | — | Exercice P5 : Gram-Schmidt sur [[1,1],[1,0]] |
| 6 | Application : similarité cosinus (textes, embeddings) | — | Exercice P6 : similarite cosinus sur vecteurs-textes |
| 7 | **Synthèse globale** (récap des 6 points) + exemple intégrateur : recherche documentaire (normes, angle, Gram-Schmidt, cosinus) | — | Exercice P7 : similarite cosinus invariante a l'echelle |

---

## T03 — Calcul différentiel & optimisation

**Fichier** : `notebooks/03_calcul_optimisation.ipynb` · **Script miroir** : `code/03_calcul_optimisation.py` · **6 points-sujets + synthèse globale / 7 exercices**

| # | Point | Sous-points | Exercice corrigé (Pn) |
|---|---|---|---|
| 1 | Dérivées & gradient (analytique et numérique) | Gradient | Exercice P1 : f'(x) de exp(-x^2), analytique vs numerique |
| 2 | Descente de gradient, convexité & Hessian | Convexité & Hessian | Exercice P2 : points critiques de g(x) = x^4 - 2x^2 |
| 3 | Régression linéaire par descente de gradient | — | Exercice P3 : pente/ordonnee par moindres carres a la main |
| 4 | Logistique, Jacobien & backpropagation `*(MML ch. 5.3-5.6)*` | Jacobien & gradients de fonctions vectorielles *(MML ch. 5.3-5.4)*<br>Backpropagation & dérivées automatiques *(MML ch. 5.6)* | Exercice P4 : gradient de f(x,y) = x^2 * y en (2,3) + gradient check |
| 5 | Hessian, Taylor & Newton `*(MML ch. 5.7-5.8)*` | — | Exercice P5 : Newton sur x^2 - 4 (x0 = 3) |
| 6 | Optimisation contrainte — Lagrange/KKT `*(MML ch. 7.2)*` | — | Exercice P6 : Lagrange min x^2 + y^2 s.t. x + y = 4 |
| 7 | **Synthèse globale** (récap + tableau « choisir l'optimiseur ») + exemple intégrateur : GD vs Newton sur x^4 - 2x^2 | — | Exercice P7 : Hessian de h(x,y) = x^2 - 2y^2 (selle) |

---

## T04 — Théorie des probabilités

**Fichier** : `notebooks/04_probabilités.ipynb` · **Script miroir** : `code/04_probabilités.py` · **6 points-sujets + synthèse globale / 7 exercices**

| # | Point | Sous-points | Exercice corrigé (Pn) |
|---|---|---|---|
| 1 | Espace probabiliste & axiomes de Kolmogorov | — | Exercice P1 : P(face paire ET pile) |
| 2 | Variables aléatoires (discrètes et continues) | Variable aléatoire continue — Normal | Exercice P2 : E[X], Var[X] d'une v.a. discrete ; P(0<=Z<=1) |
| 3 | Lois usuelles en science des données | — | Exercice P3 : P(Bin(10,0.1)>=1), P(Pois(1)>=1), E[Geom] |
| 4 | Bayes & test d'hypothèse | Bayes avec prior Bêta (conjugaison Bêta-Binomiale)<br>Test d'hypothèse (aperçu — détaillé en T06) | Exercice P4 : Bayes (test de depistage) + Poisson |
| 5 | Monte-Carlo & familles exponentielles `*(MML ch. 6.6)*` | Familles exponentielles & conjugaison *(MML ch. 6.6)* | Exercice P5 : Monte-Carlo P(|Z| < 1) |
| 6 | Changement de variable & échantillonnage inverse `*(MML ch. 6.7)*` | — | Exercice P6 : Y = -ln(X), X~U(0,1) -> Y~Exp(1) |
| 7 | **Synthèse globale** (récap des 6 points) + exemple intégrateur : Monte-Carlo exponentielle, Bayes (base rate), binomiale | — | Exercice P7 : moments de Bernoulli(p=0.7) |

---

## T05 — Modélisation statistique

**Fichier** : `notebooks/05_modelisation_statistique.ipynb` · **Script miroir** : `code/05_modelisation_statistique.py` · **6 points-sujets + synthèse globale / 7 exercices**

| # | Point | Sous-points | Exercice corrigé (Pn) |
|---|---|---|---|
| 1 | Distribution d'échantillonnage | — | Exercice P1 : loi de Xbar pour X~N(5,2), n=100 |
| 2 | Théorème de la Limite Centrale (TCL) | — | Exercice P2 : P(Xbar > 105), population N(100,15), n=9 |
| 3 | Estimation — biais & variance d'un estimateur | — | Exercice P3 : Var(Xbar, n=25) vs Var((X1+X2)/2) |
| 4 | Intervalles de confiance (IC) | — | Exercice P4 : IC95 pour une proportion (n=100, phat=0.60) |
| 5 | Régression linéaire, corrélation & R² | Corrélation & coefficient de détermination | Exercice P5a : regression a la main (5 points alignes) ; P5b : coefficient de correlation |
| 6 | Régression linéaire bayésienne `*(MML ch. 9.3)*` | — | Exercice P6 : moments de Y = X^2, X~N(0,1) (changement de variable) |
| 7 | **Synthèse globale** (récap des 6 points) + exemple intégrateur : une droite, trois regards (IC, R² = r², prior faible → OLS) | — | — |

---

## T06 — Inférence statistique & tests

**Fichier** : `notebooks/06_tests_et_inference.ipynb` · **Script miroir** : `code/06_tests_et_inference.py` · **6 points-sujets + synthèse globale / 7 exercices**

| # | Point | Sous-points | Exercice corrigé (Pn) |
|---|---|---|---|
| 1 | Tests paramétriques | — | Exercice P1 : test t 1 echantillon (n=16, Xbar=103, s=4, H0: mu=100) |
| 2 | Tests non paramétriques | — | Exercice P2 : Mann-Whitney (10+10, decalés) |
| 3 | ANOVA (Analyse de variance) | — | Exercice P3 : ANOVA (3 groupes de meme moyenne, n=30) |
| 4 | Test du chi-deux (indépendance) | — | Exercice P4 : chi-deux de conformite (de 6 faces, 300 lancers) |
| 5 | Statistique bayésienne (Bêta-Binomiale) | — | Exercice P5 : Beta-Binomiale (8/10, prior Beta(1,1)) |
| 6 | Théorie de la décision (coûts) | — | Exercice P6 : theorie de la decision (matrice de couts) |
| 7 | **Synthèse globale** (récap des 6 points) + exemple intégrateur : A/B test en z, bayésien Bêta et décision | — | Exercice P7 : p-value de z = 1.8 (deux queues) |

---

## T07 — Applications en science des données

**Fichier** : `notebooks/07_applications_datascience.ipynb` · **Script miroir** : `code/07_applications_datascience.py` · **6 points-sujets + synthèse globale / 7 exercices**

| # | Point | Sous-points | Exercice corrigé (Pn) |
|---|---|---|---|
| 1 | Pipeline de modélisation régressive (end-to-end) | — | Exercice P1 : R^2 et RMSE a la main (donnees du pipeline) |
| 2 | Validation k-fold & biais-variance | — | Exercice P2 : k-fold 5 plis sur sin |
| 3 | Sur-apprentissage explicite (biais-variance) | — | Exercice P3 : courbe de sur-apprentissage (degres 1, 3, 8, 20) |
| 4 | Classification : métriques | — | Exercice P4 : confusion matrix manuelle (12 cas, 4 positifs) |
| 5 | Biais & considérations éthiques | — | Exercice P5 : biais d'un decideur (regle "selon groupe") |
| 6 | Tableau récapitulatif — choisir la bonne méthode | — | Exercice P6 : lineaire vs arbre sur y = 3x + 2 (bruit faible) |
| 7 | **Synthèse globale** (récap des 6 points) + exemple intégrateur : pipeline déséquilibré, F1/AUC à la main vs sklearn | — | Exercice P7 : F1, precision, rappel, AUC a la main |

---

## Notebooks d'accompagnement

| Fichier | Contenu |
|---|---|
| `08_examen_preparation.ipynb` | 15 drills « papier » du DE avec corrigés pas-à-pas + vérification numérique (utilisé en S8) |
| `09_graphiques.ipynb` | 8 figures pédagogiques inline : TCL, descente de gradient, PCA, lois usuelles, Bayes, IC, sur-apprentissage, learning curves |

## Statut de vérification

- 7/7 notebooks exécutés (`nbclient`, Python 3.12, numpy 2.5, scipy 1.18, sklearn 1.9.1) — **ALL PASS**.
- 49 exercices corrigés (7 × 7), chaque valeur numérique recalculée et assertée dans la cellule.
- Colab-ready : la seule ligne d'installation est le `!pip install` du T01 (déjà présent sur Colab).