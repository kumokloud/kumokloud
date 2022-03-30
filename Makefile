.PHONY: all uplatex latex

all: main.pdf

main.pdf: main.dvi
	dvipdfmx main.dvi
	open main.pdf

main.dvi: main.tex imcreport.sty
	uplatex main.tex
	uplatex main.tex

imcreport.sty:
	cp ../template/imcreport.sty ./

