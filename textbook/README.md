# An Introduction to Statistical Learning, with Applications in Python (ISLP)

Full text converted from [`ISLP.pdf`](ISLP.pdf) to Markdown (MinerU2.5-Pro VLM, MLX backend), split into one file per chapter under [`chapters_md/`](chapters_md/) for easier navigation. Each chapter's figures live in a matching subfolder under [`images/`](images/) (e.g. `images/03-linear-regression/`). Front matter and the index have no figures. Only figures actually referenced in the text are kept — MinerU also extracts incidental sub-crops that aren't inlined anywhere, and those were dropped.

## Chapters

| # | Chapter | File |
| --- | --- | --- |
| — | Front Matter (dedication, preface, table of contents) | [chapters_md/00-front-matter.md](chapters_md/00-front-matter.md) |
| 1 | Introduction | [chapters_md/01-introduction.md](chapters_md/01-introduction.md) |
| 2 | Statistical Learning | [chapters_md/02-statistical-learning.md](chapters_md/02-statistical-learning.md) |
| 3 | Linear Regression | [chapters_md/03-linear-regression.md](chapters_md/03-linear-regression.md) |
| 4 | Classification | [chapters_md/04-classification.md](chapters_md/04-classification.md) |
| 5 | Resampling Methods | [chapters_md/05-resampling-methods.md](chapters_md/05-resampling-methods.md) |
| 6 | Linear Model Selection and Regularization | [chapters_md/06-linear-model-selection-and-regularization.md](chapters_md/06-linear-model-selection-and-regularization.md) |
| 7 | Moving Beyond Linearity | [chapters_md/07-moving-beyond-linearity.md](chapters_md/07-moving-beyond-linearity.md) |
| 8 | Tree-Based Methods | [chapters_md/08-tree-based-methods.md](chapters_md/08-tree-based-methods.md) |
| 9 | Support Vector Machines | [chapters_md/09-support-vector-machines.md](chapters_md/09-support-vector-machines.md) |
| 10 | Deep Learning | [chapters_md/10-deep-learning.md](chapters_md/10-deep-learning.md) |
| 11 | Survival Analysis and Censored Data | [chapters_md/11-survival-analysis-and-censored-data.md](chapters_md/11-survival-analysis-and-censored-data.md) |
| 12 | Unsupervised Learning | [chapters_md/12-unsupervised-learning.md](chapters_md/12-unsupervised-learning.md) |
| 13 | Multiple Testing | [chapters_md/13-multiple-testing.md](chapters_md/13-multiple-testing.md) |
| — | Index | [chapters_md/14-index.md](chapters_md/14-index.md) |

## Layout

```
textbook/
├── ISLP.pdf          original source
├── README.md         this file
├── images/           one subfolder per chapter, e.g. images/03-linear-regression/
└── chapters_md/      one file per chapter, image links point to ../images/<chapter>/
```
