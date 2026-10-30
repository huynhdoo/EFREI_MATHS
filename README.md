# EFREI_MATHS — Notebooks

Notebooks du module **ADIF72 · Mathematics for Data Scientists** (EFREI, semestre 7, 45 h).
Contenu en français ; sorties pré-exécutées dans les `.ipynb` (aucune installation requise pour la lecture).

## Notebooks

Chaque notebook T01–T07 est organisé en **6 points-sujets** numérotés, chacun avec son
**exercice corrigé Pn** intégré ; le **point 7** est réservé à la **synthèse globale**
(tableau récapitulatif des 6 points + exemple intégrateur réutilisant tous les outils). Voir le détail dans [`PLAN_NOTEBOOKS.md`](PLAN_NOTEBOOKS.md).

| # | Notebook | Contenu |
|---|---|---|
| 01 | `01_algebre_lineaire.ipynb` | Systèmes, rang/noyau, valeurs propres, SVD + eigendecomposition, Cholesky, PCA latente |
| 02 | `02_geometrie_analytique.ipynb` | Normes, angles, projections, Gram-Schmidt, similarité cosinus |
| 03 | `03_calcul_optimisation.ipynb` | Dérivées/gradients, descente de gradient, convexité + Jacobien, backpropagation, Hessian/Newton, Lagrange (KKT) |
| 04 | `04_probabilités.ipynb` | Axiomes, v.a., lois usuelles, Bayes, Monte-Carlo + famille exponentielle, changement de variable |
| 05 | `05_modelisation_statistique.ipynb` | TCL, estimation, IC, régression + régression bayésienne |
| 06 | `06_tests_et_inference.ipynb` | Tests paramétriques/non paramétriques, ANOVA+Tukey, chi², bayésien |
| 07 | `07_applications_datascience.ipynb` | Pipeline, k-fold, biais-variance, métriques, éthique |
| 08 | `08_examen_preparation.ipynb` | 15 drills « papier » du DE avec corrigés pas-à-pas |
| 09 | `09_graphiques.ipynb` | 8 figures pédagogiques (TCL, GD, PCA, lois, Bayes, IC, sur-apprentissage, learning curves) |

## Documents de référence (`docs/`)

| Fichier | Contenu |
|---|---|
| `Syllabus_ADIF72-3.pdf` | Syllabus officiel du module ADIF72 (45 h) |
| `70_concepts_mathematiques.pdf` | 70 concepts mathématiques essentiels pour la data science |

> *Note : le livre MML (Deisenroth/Faisal/Ong) n'est pas inclus (droit d'auteur) ;*
> *il est référencé dans les notebooks par ses chapitres (`*(MML ch. X.Y)*`).*

## Exécuter

- **Local** : `jupyter notebook notebooks/`
- **Google Colab** : `File → Open notebook from → Google Drive` (dossier `EFREI/MATHS/notebooks`)
- **GitHub** : `Open in Colab` (via le bouton Colab sur chaque fichier `.ipynb`)

## Dépendances
Python 3.10+ · `numpy` · `scipy` · `pandas` · `sklearn` · `matplotlib`

---
*Généré le 28/10/2026 — ADIF72 2025-2026.*
