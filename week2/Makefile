all: slides.html slides.pdf
slides.html: slides.qmd custom.scss
	quarto render slides.qmd --to revealjs
slides.pdf: slides.html
	npx --yes decktape reveal slides.html slides.pdf
preview:
	quarto preview slides.qmd
clean:
	rm -rf slides_files slides.html slides.pdf
