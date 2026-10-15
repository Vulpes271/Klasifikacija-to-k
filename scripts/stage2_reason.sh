#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_stage.sh" stage2_reason "$@"
