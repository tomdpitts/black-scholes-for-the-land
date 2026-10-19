#!/bin/sh
# Build main.pdf and supplement.pdf with pdfLaTeX (AAAI-27 requires it).
export PATH=$PATH:$HOME/Library/TinyTeX/bin/universal-darwin
cd "$(dirname "$0")"
for f in main supplement; do pdflatex -interaction=nonstopmode $f.tex > /dev/null; done
for f in main supplement; do bibtex $f > /dev/null; done
for i in 1 2; do for f in main supplement; do pdflatex -interaction=nonstopmode $f.tex > $f.build.log; done; done
grep -hE "^!|Overfull|undefined|Output written" main.build.log supplement.build.log
