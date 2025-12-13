.PHONY: build-docs
build-docs:
	rm -rf build
	plantuml -o ../build/ docs/**.puml
	(cd docs && xelatex -output-directory=../build --halt-on-error checkpoint1_report.tex)
	(cd docs && xelatex -output-directory=../build --halt-on-error checkpoint2_report.tex)
