# ML Summer Class

A 7-week summer course covering the foundations of statistical / machine learning, paired with hands-on Python labs from *An Introduction to Statistical Learning*.

## Syllabus

| | |
|---|---|
| **Instructors** | Ho-min Park and Hwan |
| **Format** | Weekly sessions, Fridays 2:00–4:00 PM Central, for 7 weeks |
| **Dates** | Jul 10 – Aug 21, 2026 |
| **Textbook** | *An Introduction to Statistical Learning, with Applications in Python* (ISLP) — James, Witten, Hastie, Tibshirani, Taylor |

## Requirements

- Read the assigned chapter(s) **before** each session — given the condensed pace, class time is focused on discussion and the Python labs rather than lecture.
- A working Python environment (e.g. via `conda` or `venv`) capable of running the ISLP labs (`jupyter`, `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `ISLP` package).

## Textbook

The course textbook is [*An Introduction to Statistical Learning, with Applications in Python*](https://www.statlearning.com/) (ISLP), by James, Witten, Hastie, Tibshirani, and Taylor.

- A copy is included in this repository: [`textbook/ISLP.pdf`](textbook/ISLP.pdf), obtained via the authors' official free-download page ([statlearning.com](https://www.statlearning.com/)).
- The instructor-provided share link is also available [here](https://go.bioinfolder.com/web/client/pubshares/ghc2TFtQDQefKPu76kH2Dt/download?compress=false).

## Schedule

| # | Date | Topics | Chapters | Reading |
|---|------|--------|----------|---------|
| 1 | Jul 10 | Intro + Statistical Learning basics | Ch 1–2 | [`week1/`](week1/) |
| 2 | Jul 17 | Linear Regression; Classification (logistic regression, LDA, QDA, KNN) | Ch 3–4 | [`week2/`](week2/) |
| 3 | Jul 24 | Resampling (Cross-Validation & Bootstrap); Regularization (subset selection, ridge, lasso) | Ch 5–6 | [`week3/`](week3/) |
| 4 | Jul 31 | Moving Beyond Linearity (splines, GAMs); Tree-Based Methods | Ch 7–8 | [`week4/`](week4/) |
| 5 | Aug 7 | Support Vector Machines; Deep Learning I (neural network basics, CNNs) | Ch 9, Ch 10 (part 1) | [`week5/`](week5/) |
| 6 | Aug 14 | Deep Learning II (RNNs, network tuning); Unsupervised Learning | Ch 10 (part 2), Ch 12 | [`week6/`](week6/) |
| 7 | Aug 21 | Survival Analysis; Multiple Testing; wrap-up | Ch 11, Ch 13 | [`week7/`](week7/) |

Each session pairs conceptual material with the corresponding Python lab(s) from the textbook.

## Weekly readings

The `week1/` – `week7/` directories each contain the chapter excerpt(s) (split
from `textbook/ISLP.pdf`) relevant to that week's session, so you can read
just the assigned unit without downloading the whole book:

- `week1/ch01-02-intro-statistical-learning.pdf` — Ch 1–2
- `week2/ch03-04-regression-classification.pdf` — Ch 3–4
- `week3/ch05-06-resampling-regularization.pdf` — Ch 5–6
- `week4/ch07-08-nonlinearity-trees.pdf` — Ch 7–8
- `week5/ch09-10pt1-svm-deep-learning-intro.pdf` — Ch 9, Ch 10.1–10.4
- `week6/ch10pt2-rnn-deep-learning.pdf` — Ch 10.5–10.10
- `week6/ch12-unsupervised-learning.pdf` — Ch 12
- `week7/ch11-survival-analysis.pdf` — Ch 11
- `week7/ch13-multiple-testing.pdf` — Ch 13

## Slide decks

Session slides are built with [Quarto](https://quarto.org) (`slides.qmd` →
revealjs); see [`AGENTS.md`](AGENTS.md) for the deck conventions used across
the course.

| Date | Title | Description | Slides |
|---|---|---|---|
| — | Reading the Math | Notation primer — scalars/vectors/matrices, sub/superscripts, Greek letters, and simple linear regression worked three ways; reference deck for the whole course | [`week0-notation/slides.qmd`](week0-notation/slides.qmd) · [`slides.pdf`](week0-notation/slides.pdf) |
| Jul 10 | Why Statistical Learning? | Motivation, importance, and how statistical learning works — Week 1 kickoff | [`week1/slides.qmd`](week1/slides.qmd) · [`slides.pdf`](week1/slides.pdf) |
| Jul 17 | Linear Regression & Classification | Least squares, multiple regression, logistic regression, LDA/QDA, ROC — Week 2 | [`week2/slides.qmd`](week2/slides.qmd) · [`slides.pdf`](week2/slides.pdf) |
| Jul 24 | Trusting and Taming Your Models | Validation, LOOCV & k-fold CV, the bootstrap, subset selection, ridge & lasso — Week 3 | [`week3/slides.qmd`](week3/slides.qmd) · [`slides.pdf`](week3/slides.pdf) |
| Jul 31 | Beyond Straight Lines: Splines & Trees | Polynomials, splines, GAMs, decision trees, bagging, random forests, boosting — Week 4 | [`week4/slides.qmd`](week4/slides.qmd) · [`slides.pdf`](week4/slides.pdf) |
| Aug 7 | Margins and Layers | Maximal margin, support vector classifiers & kernels, neural networks, CNNs — Week 5 | [`week5/slides.qmd`](week5/slides.qmd) · [`slides.pdf`](week5/slides.pdf) |
| Aug 14 | Sequences and Structure | RNNs & embeddings, fitting networks (SGD, dropout, double descent), PCA, K-means & hierarchical clustering — Week 6 | [`week6/slides.qmd`](week6/slides.qmd) · [`slides.pdf`](week6/slides.pdf) |
| Aug 21 | How Long, and How Sure? | Kaplan–Meier, Cox proportional hazards, FWER & FDR, Benjamini–Hochberg, course wrap-up — Week 7 | [`week7/slides.qmd`](week7/slides.qmd) · [`slides.pdf`](week7/slides.pdf) |

## Labs (Python notebooks)

Each week's session includes the official ISLP lab notebook(s), copied from
[intro-stat-learning/ISLP_labs](https://github.com/intro-stat-learning/ISLP_labs)
into the corresponding week folder so everything needed for a session lives in
one place. Install the dependencies once with
`pip install -r lab-requirements.txt`.

| Week | Lab(s) | Notebooks |
|---|---|---|
| 1 | 2.3 Introduction to Python | [`week1/Ch02-statlearn-lab.ipynb`](week1/Ch02-statlearn-lab.ipynb) |
| 2 | 3.6 Linear Regression · 4.7 Classification | [`Ch03`](week2/Ch03-linreg-lab.ipynb) · [`Ch04`](week2/Ch04-classification-lab.ipynb) |
| 3 | 5.3 CV & Bootstrap · 6.5 Regularization | [`Ch05`](week3/Ch05-resample-lab.ipynb) · [`Ch06`](week3/Ch06-varselect-lab.ipynb) |
| 4 | 7.8 Non-Linear Modeling · 8.3 Trees | [`Ch07`](week4/Ch07-nonlin-lab.ipynb) · [`Ch08`](week4/Ch08-baggboost-lab.ipynb) |
| 5 | 9.6 SVM · 10.9 Deep Learning (part 1) | [`Ch09`](week5/Ch09-svm-lab.ipynb) · [`Ch10`](week5/Ch10-deeplearning-lab.ipynb) |
| 6 | 10.9 Deep Learning (part 2) · 12.5 Unsupervised | [`Ch10`](week6/Ch10-deeplearning-lab.ipynb) · [`Ch12`](week6/Ch12-unsup-lab.ipynb) |
| 7 | 11.8 Survival Analysis · 13.6 Multiple Testing | [`Ch11`](week7/Ch11-surv-lab.ipynb) · [`Ch13`](week7/Ch13-multiple-lab.ipynb) |

## Homework (Excel worksheets)

Each week has a hands-on Excel homework in the spirit of the
[`excel_worksheets/`](excel_worksheets/) collection: small, hand-computable
datasets, formulas entered step by step in the yellow cells, instant ✓/✗
auto-checking, and an `Answer_Key` sheet. Every workbook's Instructions sheet
lists the matching ISLP sections and slide titles.

| Week | Topics | File |
|---|---|---|
| 1 | Train/test MSE, bias–variance decomposition, KNN by hand | [`week1/homework-test-error-knn.xlsx`](week1/homework-test-error-knn.xlsx) |
| 2 | Least squares fit, RSE/R²/t-test, logistic predictions, confusion matrix | [`week2/homework-regression-classification.xlsx`](week2/homework-regression-classification.xlsx) |
| 3 | k-fold CV, bootstrap SE, ridge shrinkage path, lasso soft-threshold | [`week3/homework-cv-bootstrap-shrinkage.xlsx`](week3/homework-cv-bootstrap-shrinkage.xlsx) |
| 4 | Step functions, greedy tree splits by RSS, Gini, bagging & boosting | [`week4/homework-step-functions-trees.xlsx`](week4/homework-step-functions-trees.xlsx) |
| 5 | Hyperplane classification, hinge loss, radial kernel, NN forward pass | [`week5/homework-svm-neural-net.xlsx`](week5/homework-svm-neural-net.xlsx) |
| 6 | K-means iterations, complete-linkage fusion, PVE, tiny RNN unroll | [`week6/homework-clustering-pca-rnn.xlsx`](week6/homework-clustering-pca-rnn.xlsx) |
| 7 | Kaplan–Meier estimation, hazard ratios, Bonferroni/Holm/BH | [`week7/homework-survival-multiple-testing.xlsx`](week7/homework-survival-multiple-testing.xlsx) |
