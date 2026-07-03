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
| Jul 10 | Why Statistical Learning? | Motivation, importance, and how statistical learning works — Week 1 kickoff | [`week1/slides.qmd`](week1/slides.qmd) (run `make` in `week1/` to build `slides.pdf`) |
