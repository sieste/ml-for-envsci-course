slides: slides.pdf

slides.pdf: slides.tex
	pdflatex slides.tex
clean:
	rm -f slides.nav slides.log slides.out slides.pdf slides.snm slides.toc

