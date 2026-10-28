#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
report_dir=${1:-reports/THUD_VMESNO}
source_name=${2:-porocilo.tex}
report_path=$(realpath "$report_dir")
if command -v pdflatex >/dev/null; then
  (cd "$report_path" && pdflatex -interaction=nonstopmode -halt-on-error "$source_name" && pdflatex -interaction=nonstopmode -halt-on-error "$source_name")
else
  if ! docker image inspect thud-latex:local >/dev/null 2>&1; then
    docker build -t thud-latex:local -f reports/THUD_VMESNO/Dockerfile.latex reports/THUD_VMESNO
  fi
  for pass in 1 2; do
    docker run --rm --user "$(id -u):$(id -g)" -v "$report_path:/report" \
      --entrypoint pdflatex thud-latex:local -interaction=nonstopmode -halt-on-error "$source_name"
  done
fi
