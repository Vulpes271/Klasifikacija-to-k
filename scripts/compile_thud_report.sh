#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v pdflatex >/dev/null; then
  (cd reports/THUD_VMESNO && pdflatex -interaction=nonstopmode -halt-on-error porocilo.tex && pdflatex -interaction=nonstopmode -halt-on-error porocilo.tex)
else
  docker build -t thud-latex:local -f reports/THUD_VMESNO/Dockerfile.latex reports/THUD_VMESNO
  for pass in 1 2; do
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/reports/THUD_VMESNO:/report" thud-latex:local
  done
fi
