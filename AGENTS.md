# AGENTS.md

Guidance for coding agents working in this repo.

## Repo shape

This is a flat collection of standalone presentation decks — one directory
per talk/tutorial, each self-contained (own figures, own build config). There
is no shared build system or root-level Quarto project (no `_quarto.yml`);
each deck renders independently. If the repo has a root `README.md` with an
index table of talks, add a row (Date / Title / Description / Link to
`slides.pdf`) for any new deck.

Decks are built with **Quarto** (`slides.qmd` → revealjs). This file
documents that pattern.

## Quarto deck anatomy

Each deck directory contains:

- `slides.qmd` — the deck content and YAML frontmatter.
- `custom.scss` — a **per-deck** SCSS theme. Themes are not shared between
  decks; each one has its own copy, tweaked for that talk. Check an existing
  deck's `custom.scss` for reusable patterns rather than assuming a common
  stylesheet exists.
- `Makefile`:
  ```
  all: slides.html slides.pdf
  slides.html: slides.qmd custom.scss
      quarto render slides.qmd --to revealjs
  slides.pdf: slides.html
      npx --yes decktape reveal slides.html slides.pdf
  preview:
      quarto preview slides.qmd
  clean:
      rm -rf slides_files slides.html slides.pdf
  ```
- `images/` or `figures/` — figure assets referenced from the `.qmd`.

### Build / preview

```bash
cd <deck-dir>
make preview   # live-reload local preview (quarto preview)
make           # render slides.html, then slides.pdf via decktape
make clean     # remove generated slides.html / slides.pdf / slides_files
```

Requires the `quarto` CLI and Node/`npx` (for `decktape`) on PATH — neither
is guaranteed to be installed in a given sandbox; check with
`quarto --version` before assuming a render will work, and say so if it
can't be verified.

Generated output (`slides.html`, `slides_files/`, sometimes `slides.pdf`) may
be checked in or gitignored depending on the deck — check `.gitignore` and
match whatever that deck already does rather than changing its policy.

### YAML frontmatter conventions

```yaml
---
title: "..."
subtitle: "..."
author: "..."
date: YYYY-MM-DD
format:
  revealjs:
    theme: [default, custom.scss]
    width: 1920
    height: 1080
    slide-number: c/t
    center: false
    transition: fade
    footer: "..."               # optional
    highlight-style: github     # if the deck has code blocks
    code-line-numbers: false
    include-in-header:
      - text: |
          <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    title-slide-attributes:      # optional branded title slide
      data-background-image: "images/....png"
      data-background-size: contain
---
```

### Slide markup conventions

- `#` (H1) = section-divider slide, styled with a custom class + solid
  background, e.g. `# Section title {.divider background-color="#123456"}`.
- `##` (H2) = a normal content slide (one H2 = one slide).
- Per-slide background image: `{background-image="..." background-size="contain"}`
  on the heading.
- Speaker notes: a `::: notes` (or `:::: notes`) fenced div at the end of a
  slide's content — not shown during presentation, visible in speaker view.
- Multi-column layout: `:::: {.columns}` / `::: {.column width="40%"}`.
- Emphasis/inline styling uses custom spans/classes defined in that deck's
  `custom.scss` (e.g. `[text]{.hl}`, `[text]{.highlight}`, `{.cite}`) rather
  than plain Markdown emphasis — check the deck's `custom.scss` for the class
  vocabulary before inventing a new one.
- Raw HTML blocks (`` ```{=html} ``) are used for bespoke diagrams or layouts
  that go beyond what Markdown/Pandoc divs can express.

### `custom.scss` conventions

- `/*-- scss:defaults --*/` section: brand color variables (`$...`) and
  Reveal.js Sass variable overrides (`$presentation-heading-font`,
  `$presentation-heading-color`, `$body-color`, `$presentation-font-size-root`,
  etc).
- `/*-- scss:rules --*/` section: plain CSS rules scoped under
  `.reveal .slides section...`, plus custom component classes for that
  specific deck's layout needs.
- Decks commonly pull a web font (e.g. Google Fonts) via `@import url(...)`.

## Adding a new Quarto deck

1. Create a new top-level directory (short, kebab/underscore name matching
   existing convention).
2. Copy the `Makefile` verbatim from an existing deck.
3. Start `slides.qmd` from the frontmatter template above; start
   `custom.scss` from the simplest existing deck rather than a heavily
   bespoke/branded one, unless the talk needs pixel-matched branded layouts.
4. Put figures under `images/` or `figures/` inside the new directory.
5. Update the root `README.md` index (if present) once `slides.pdf` exists.
6. Run `make preview` (or `make`) to verify the deck renders before
   considering the work done — don't just eyeball the `.qmd`.
