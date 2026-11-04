#!/usr/bin/env bash
# ARHIVSKI IZVAJALNIK: samo reprodukcija objavljenih poskusov; aktivni ukaz je v README.md.
set -euo pipefail
exec bash "$(dirname "$0")/run_stage.sh" stage2_reason "$@"
