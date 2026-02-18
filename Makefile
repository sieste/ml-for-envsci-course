slides: slides.pdf
archive: ml4env-course.zip

slides.pdf: slides.tex
	pdflatex slides.tex

ml4env-course.zip: 
	mkdir /tmp/ml4env-course
	cp slides.pdf code -r /tmp/ml4env-course
	zip -r ml4env-course.zip /tmp/ml4env-course
	rm -rf /tmp/ml4env-course


clean:
	rm -f slides.nav slides.log slides.out slides.pdf slides.snm slides.toc

