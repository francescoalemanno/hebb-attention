#!/usr/bin/env bash
# Build the arXiv source package of main.tex.
#
#   1. compile main.tex (pdflatex, bibtex, pdflatex x2);
#   2. collect main.tex (with the bibliography inlined in place of \bibliography) and the figures it includes into arxiv_source.zip;
#   3. unpack the zip in an empty temporary folder and compile it there with pdflatex only,
#      as arXiv does, and check that it gives the same text as main.pdf.
#
# Usage: ./build_arxiv.sh   (from anywhere; it works in the folder of this script)
set -euo pipefail
cd "$(dirname "$0")"

stage=$(mktemp -d); test_dir=$(mktemp -d)
trap 'rm -rf "$stage" "$test_dir"' EXIT

run_pdflatex() {  # compile quietly, stop on LaTeX errors
    if ! pdflatex -interaction=nonstopmode -halt-on-error "$1" > /dev/null; then
        grep -A3 '^!' "${1%.tex}.log" >&2 || true
        echo "pdflatex failed on $PWD/$1" >&2; exit 1
    fi
}

echo "== compiling main.tex"
run_pdflatex main.tex
bibtex main > /dev/null || { cat main.blg >&2; exit 1; }
run_pdflatex main.tex
run_pdflatex main.tex
if grep -qE 'undefined|Rerun to get' main.log; then
    grep -E 'undefined|Rerun to get' main.log >&2; echo "unresolved references in main.tex" >&2; exit 1
fi

echo "== collecting sources"
# the bibliography is inlined in the staged main.tex: the package carries neither main.bbl nor the .bst
python3 - "$stage" <<'PY'
import re, sys, pathlib
src = pathlib.Path("main.tex").read_text(); bbl = pathlib.Path("main.bbl").read_text().strip()
new, n = re.subn(r"\\bibliographystyle\{[^}]*\}\s*\n\\bibliography\{[^}]*\}", lambda m: bbl, src)
assert n == 1, "bibliography commands not found in main.tex"
pathlib.Path(sys.argv[1], "main.tex").write_text(new)
PY
figs=$(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' main.tex | sed -E 's/.*\{([^}]*)\}/\1/' | sort -u)
for f in $figs; do
    mkdir -p "$stage/$(dirname "$f")"
    cp "$f" "$stage/$f"
done
rm -f arxiv_source.zip
(cd "$stage" && zip -q -X -r "$OLDPWD/arxiv_source.zip" .)
echo "   $(echo "$figs" | wc -l | tr -d ' ') figures, written arxiv_source.zip"

echo "== test build in an empty folder"
unzip -q arxiv_source.zip -d "$test_dir"
(cd "$test_dir" && run_pdflatex main.tex && run_pdflatex main.tex)
if grep -qE 'undefined|not found' "$test_dir/main.log"; then
    grep -E 'undefined|not found' "$test_dir/main.log" >&2; echo "the package does not compile cleanly" >&2; exit 1
fi
if command -v pdftotext > /dev/null; then
    if cmp -s <(pdftotext "$test_dir/main.pdf" -) <(pdftotext main.pdf -); then
        echo "   same text as main.pdf"
    else
        echo "the package gives a different text from main.pdf" >&2; exit 1
    fi
fi
pages=$(pdfinfo "$test_dir/main.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
echo "   compiles on its own${pages:+ ($pages pages)}"

rm -f main.aux main.log main.out main.blg main.bbl
echo "== done: arxiv_source.zip"
