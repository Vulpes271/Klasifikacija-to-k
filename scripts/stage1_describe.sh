#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_stage.sh" stage1_describe "$@"
