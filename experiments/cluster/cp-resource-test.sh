#!/usr/bin/env bash
set -euo pipefail
HP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$HP_SCRIPT_DIR/resource-test.sh" cp_sat "$@"
