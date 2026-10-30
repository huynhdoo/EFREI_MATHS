# Plan d'étude — ADIF72 · Mathematics for Data Scientists (2025-2026)

> Base : `Syllabus_ADIF72-3.pdf`. Objectif : couvrir le programme en 45 h
> (30 h CTD + 15 h personnel) de façon progressive, avec de la théorie,
> des exemples et des exercices corrigés en Python (`code/`).

## Prérequis vérifiés
Python 3.12 avec `numpy`, `scipy`, `pandas`, `sklearn`, `matplotlib`.
Avant de commencer, réviser : bases d'algèbre linéaire, géométrie, stats
descriptives, et savoir manipuler des tableaux numpy (`np.array`,
`reshape`, indexation, opérations vectorisées).

## Jupyter Notebooks (`notebooks/`)
9 notebooks complets, avec explications détaillées et sorties exécutées
(outputs pré-calculés dans le .ipynb) — disponibles sur **Google Colab**
(`File → Open notebook from → Google Drive`, dossier `EFREI/MATHS/notebooks`)
ou JupyterLab local (`jupyter notebook notebooks/`).

| Notebook | Contenu (6 points-sujets + synthèse · MML book) |
|---|---|
| `01_algebre_lineaire.ipynb` | T01 — 6 pts + synthèse : opérations/systèmes, rang & bases, valeurs propres, SVD, PCA+eigendecompo+Cholesky *(ch. 4.3–4.4)*, PCA latente *(ch. 10.7)*, synthèse |
| `02_geometrie_analytique.ipynb` | T02 — 6 pts + synthèse : normes, produit scalaire/angle, projection, distances, Gram-Schmidt, similarité cosinus, synthèse |
| `03_calcul_optimisation.ipynb` | T03 — 6 pts + synthèse : dérivées/gradient, GD+convexité/Hessian, régression GD, logistique+Jacobien+backprop *(ch. 5.3–5.6)*, Hessian/Taylor/Newton *(ch. 5.7–5.8)*, Lagrange/KKT *(ch. 7.2)*, synthèse globale (tableau optimiseurs) |
| `04_probabilités.ipynb` | T04 — 6 pts + synthèse : Kolmogorov, v.a. discrète/continue, lois usuelles, Bayes/test, Monte-Carlo+famille exponentielle *(ch. 6.6)*, changement de variable *(ch. 6.7)*, synthèse |
| `05_modelisation_statistique.ipynb` | T05 — 6 pts + synthèse : échantillonnage, TCL, estimation, IC, régression + corrélation/R², régression bayésienne *(ch. 9.3)* |
| `06_tests_et_inference.ipynb` | T06 — 6 pts + synthèse : paramétriques, non paramétriques, ANOVA+Tukey, chi², bayésien Bêta, théorie de décision, synthèse globale (lecture p-value) |
| `07_applications_datascience.ipynb` | T07 — 6 pts + synthèse : pipeline end-to-end, k-fold/biais-variance, sur-apprentissage, métriques, biais & éthique, récap méthode, synthèse globale (exemple F1/AUC) |
| `08_examen_preparation.ipynb` | 15 drills "papier" du DE avec corrigés pas-à-pas + vérification |
| `09_graphiques.ipynb` | 8 figures pédagogiques inline (TCL, GD, PCA, lois, Bayes, IC, sur-appr., learning curves) |

Chaque notebook T01–T07 comporte **6 points-sujets** numérotés `## 1 ·` … `## 6 ·`,
chacun contenant son **exercice corrigé Pn** (bloc de code `# ---- Exercice Pn ----`).
Le point `## 7 ·` est réservé à la **synthèse globale** : tableau récapitulatif des
6 points + exemple intégrateur qui réutilise tous les outils du notebook.
Les ajouts MML (livre Deisenroth/Faisal/Ong, `mml-book.com`) comblent les
gaps entre le syllabus ADIF72 et le livre de référence : les points
concernés sont signalés par un badge `*(MML ch. X.Y)*` dans les notebooks.
Non couvert (intentionnellement) : GMM/EM (ch. 11), SVM (ch. 12),
graphical models (ch. 8.4) — hors périmètre ADIF72.

## Organisation en 8 séquences de 3 h 30
Chaque séquence = **1 CTD de 3 h 30** (le cours entier = 28 h face-à-face).
8 séquences couvrent les 7 topics (T01–T07) + 1 session de préparation examen.
Rythme interne type (à adapter selon le topic) :
- **00:00–00:15** (15 min) — raccroche + objectifs de la session
- **00:15–02:30** (135 min) — points 1–6 du notebook : théorie + exemples exécutés,
  **exercice Pn corrigé intégré dans chaque point n** (à refaire sans regarder)
- **02:30–03:00** (30 min) — point 7 (synthèse/récap du topic)
- **03:00–03:30** (30 min) — drill papier (type DE) + auto-évaluation

> Matériel par séquence : 1 notebook (`notebooks/`) + 1 script (`code/`) du même
> numéro, exécutés ensemble. Les sections signalées `*(MML ch. X.Y)*` correspondent
> au bloc de 45 min.

### S1 · T01 — Algèbre linéaire
**Objectifs :** résoudre des systèmes, maîtriser rang/noyau, valeurs propres, SVD, PCA.
- 00:00–00:15 · Rappel + pourquoi l'algèbre matricielle est le socle du ML
- 00:15–01:30 · Points 1–3 : opérations/systèmes, rang & bases, valeurs propres
- 01:30–02:30 · Points 4–5 : SVD, PCA/eigendecomposition/Cholesky *(MML ch. 4.3–4.4)*
- 02:30–03:00 · Point 6 : PCA perspective variable latente *(MML ch. 10.7)*
- 03:00–03:30 · Point 7 (synthèse) + drill papier A1/A2/A3 — *DE Partie A*
> **Exercices P1–P6 + l'exercice de la synthèse globale** : intégrés dans chaque point du notebook (à refaire sans regarder).

### S2 · T02 — Géométrie analytique
**Objectifs :** normes, angles, projections, bases orthonormales, similarité.
- 00:00–00:15 · Rappels S1 (produit matriciel, orthogonalité)
- 00:15–01:30 · Points 1–3 : normes L1/L2/L∞, produit scalaire/angle, projection
- 01:30–02:30 · Points 4–5 : distances, Gram-Schmidt (bases orthonormales)
- 02:30–03:00 · Point 6 : similarité cosinus (recherche, embeddings) — lien SVD
- 03:00–03:30 · Point 7 (synthèse distances/similarités) + drill papier A4/A5
> **Exercices P1–P6 + l'exercice de la synthèse globale** : intégrés dans chaque point du notebook.

### S3 · T03 — Calcul différentiel & optimisation
**Objectifs :** dérivées/gradients, descente de gradient, convexité, Lagrange.
- 00:00–00:15 · Le calcul comme moteur de l'entraînement des modèles
- 00:15–01:30 · Points 1–3 : dérivées, descente de gradient, convexité & Hessian
- 01:30–02:30 · Points 4–5 : régression linéaire/logistique + Jacobien/backprop *(MML ch. 5.3–5.8)*
- 02:30–03:00 · Point 6 : optimisation contrainte Lagrange/KKT *(MML ch. 7.2)*
- 03:00–03:30 · Point 7 (récap "quel optimiseur") + drill papier B1/B2/B3
> **Exercices P1–P6 + l'exercice de la synthèse globale** : intégrés dans chaque point du notebook.

### S4 · T04 — Théorie des probabilités
**Objectifs :** espaces probabilistes, v.a., lois usuelles, Bayes, incertitude.
- 00:00–00:15 · Le pont entre calcul et statistique
- 00:15–01:30 · Points 1–2 : axiomes de Kolmogorov, v.a. discrète/continue
- 01:30–02:30 · Points 3–4 : lois usuelles, Bayes (paradoxe du taux de base)
- 02:30–03:00 · Points 5–6 : Monte-Carlo/famille exponentielle *(MML ch. 6.6–6.7)*
- 03:00–03:30 · Point 7 (synthèse) + drill papier B4/B5/B6 (Bayes, normale, binomiale)
> **Exercices P1–P6 + l'exercice de la synthèse globale** : intégrés dans chaque point du notebook.

### S5 · T05 — Modélisation statistique
**Objectifs :** TCL, estimation (biais/variance), intervalles de confiance, régression.
- 00:00–00:15 · D'une population à un échantillon
- 00:15–01:30 · Points 1–3 : distribution d'échantillonnage, TCL, estimation
- 01:30–02:30 · Points 4–5 : intervalles de confiance, régression linéaire
- 02:30–03:00 · Point 5 (fin) : corrélation/R² + Point 6 : régression bayésienne *(MML ch. 9.3)*
- 03:00–03:30 · Drill papier C1/C2/C4 (moyenne/var, IC, régression + R²)
> **Exercices P1–P6 + l'exercice de la synthèse globale** : intégrés dans chaque point du notebook.

### S6 · T06 — Inférence & tests
**Objectifs :** tests paramétriques/non paramétriques, ANOVA, chi², décision.
- 00:00–00:15 · La procédure de décision (H0/H1, p-value, α)
- 00:15–01:30 · Points 1–3 : tests paramétriques (z, t, binomial), non paramétriques, ANOVA
- 01:30–02:30 · Points 4–5 : chi² indépendance, statistique bayésienne Bêta
- 02:30–03:00 · Points 6–7 : théorie de décision + synthèse "lecture d'une p-value"
- 03:00–03:30 · Drill papier C3/C5/C6 (test t, ANOVA F, chi²)
> **Exercices P1–P6 + l'exercice de la synthèse globale** : intégrés dans chaque point du notebook.

### S7 · T07 — Applications en science des données
**Objectifs :** pipeline, validation, biais-variance, métriques, éthique.
- 00:00–00:15 · Assembler T01–T06 dans un vrai pipeline
- 00:15–01:30 · Points 1–3 : pipeline régressif end-to-end, k-fold, sur-apprentissage
- 01:30–02:30 · Points 4–5 : métriques de classification, biais & éthique
- 02:30–03:00 · Points 6–7 : tableau récap "quelle méthode" + synthèse F1/AUC
- 03:00–03:30 · Revue transversale : relier SVD→PCA→corrélation→tests ; préparer le TD (40 %)
> **Exercices P1–P6 + l'exercice de la synthèse globale** : intégrés dans chaque point du notebook.

### S8 · Préparation examen (DE + TD)
**Objectifs :** consolider, s'entraîner au papier, simuler.
- 00:00–00:30 · Relectures ciblées des 15 drills du notebook 08 (sans exécuter)
- 00:30–01:30 · **Simulation blanche #1 (60 min, papier)** : partie A+B+C
  (utiliser `exams/devoir_1.md`)
- 01:30–02:00 · Correction + repérage des points faibles (`exams/corrige_devoir_1.md`)
- 02:00–03:00 · **Simulation blanche #2 (60 min, papier)** :
  `exams/devoir_2.md` + entraînement or TD (groupe)
- 03:00–03:30 · Simulation graphique (notebook 09) + planning de révision finale

> **Les 4 dévoirs blancs** (`exams/devoir_{1..4}.md`, chacun 2 h · 20 points ·
> format DE, corrigés vérifiés dans `exams/corrige_devoir_{1..4}.md`) sont à
> utiliser en hors-séquence pour l'entraînement papier : 1 devoir blanc par
> semaine minimum, à commencer au plus tard à la S7. Objectif : ≥ 14/20 sur
> au moins 2 dévoirs avant le jour J.

## Barème et stratégie d'examen
- **Devoir écrit (DE)** : individuel, 2 h, papier, **60 %**.
- **Interrogation de TD** : groupe, **40 %**.
  → Préparer des exemples à expliquer oralement en binôme/trio.
  → Le DE se joue sur la maîtrise du calcul manuel (dérivées, proba,
     tests) ; le TD sur la capacité à présenter/justifier.

## Bibliographie (ordre de lecture conseillé)
1. Deisenroth, Faisal, Ong — *Mathematics for Machine Learning*
   (gratuit en ligne ; T01-T04).
2. Lay — *Algèbre Linéaire et Ses Applications* (T01-T02).
3. Sheldon Ross — *Intro à la Probabilité et aux Stats* (T04-T06).
4. Casella & Berger — *Statistical Inference* (T05-T06).

## Conventions des fichiers python
- Code avec docstrings et commentaires en français.
- Chaque fichier est **auto-éxécutable** (`python3 code/0X_....py`)
  et affiche les résultats + des assertions qui vérifient les calculs.
- Les exercices sont numérotés ; le corrigé est inline avec le bloc
  `# ---- EXERCICE n°... (CARRIGÉ) ----`.

## Suivi de progression
- [ ] S1 · T01 Algèbre linéaire
- [ ] S2 · T02 Géométrie analytique
- [ ] S3 · T03 Calcul différentiel et optimisation
- [ ] S4 · T04 Théorie des probabilités
- [ ] S5 · T05 Modélisation statistique
- [ ] S6 · T06 Tests et inférence
- [ ] S7 · T07 Applications en science des données
- [ ] S8 · Préparation examen (2 simulations blanches + or TD)

## Dévoirs blancs (`exams/`)
4 dévoirs complets au format DE (2 h · 20 points · sans document ·
A 6 pt / B 6 pt / C 8 pt), chacun avec son corrigé détaillé.
Toutes les réponses corrigées ont été **vérifiées numériquement**
(numpy/scipy) avant mise en ligne.

| Devoir | Part A (algèbre) | Part B (calcul/proba) | Part C (stats/tests) |
|---|---|---|---|
| `devoir_1.md` | système 3×3, det/inverse, VP P, angle, projection | x³−3x extrema, GD f(x), Bayes test, N(60,10) | stats échantillon, IC t, test t 2 pop, régression R² |
| `devoir_2.md` | A·B, rang/noyau, Gram-Schmidt, rotation 90°, Cholesky | chaîne e^{x²}, Lagrange x+y=4, Poisson, CLT X̄ | CLT, IC proportion, ANOVA 3 groupes, chi² 2×2 |
| `devoir_3.md` | x∈solution + det, inverse N, A²=I, normes ‖·‖, projection | Newton √4, min quadratique, 3 usines Bayes, Exp mémoire | efficacité estimateurs, IC σ connu, test z proportion, régression |
| `devoir_4.md` | det/inverse A, VP √5, cos θ, plan n·X, C⁻¹ 3×3 | Exp(1) + médiane, gradient = g, mode/médiane/moyenne, quantiles | test z défauts, chi² ville/rural, Welch 2 pop, régression R²=1 |

Corrigés : `corrige_devoir_1.md` … `corrige_devoir_4.md`.
