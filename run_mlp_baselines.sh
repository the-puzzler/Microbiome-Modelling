#!/usr/bin/env bash
# Run MLP baselines for gingivitis, snowmelt, and DIABIMMUNE from repo root
# Usage: bash run_mlp_baselines.sh

set -u  # treat unset variables as an error

# Where to store all stdout/stderr
LOG_FILE="mlp_baselines_log_$(date +%Y%m%d_%H%M%S).txt"

# Use non-interactive backend so matplotlib doesn't try to open windows
export MPLBACKEND=Agg

echo "Writing all MLP baseline outputs to: $LOG_FILE"
echo "Started at: $(date)" > "$LOG_FILE"
echo >> "$LOG_FILE"

run_step() {
  local name="$1"; shift
  echo "===== $name =====" | tee -a "$LOG_FILE"
  echo "Command: $*"       | tee -a "$LOG_FILE"
  "$@" >>"$LOG_FILE" 2>&1
  local status=$?
  if [ $status -eq 0 ]; then
    echo "[OK] $name" | tee -a "$LOG_FILE"
  else
    echo "[FAILED] $name (exit $status)" | tee -a "$LOG_FILE"
  fi
  echo >> "$LOG_FILE"
}

# Gingivitis MLP baselines (colonisation + dropout)
run_step "Gingivitis MLP baselines" \
  uv run python -u scripts/gingivitis/base_lines_mlp.py

# Snowmelt MLP baselines (colonisation + dropout)
run_step "Snowmelt MLP baselines" \
  uv run python -u scripts/snowmelt/base_lines_mlp.py

# DIABIMMUNE MLP baselines (colonisation + dropout)
run_step "DIABIMMUNE MLP baselines" \
  uv run python -u scripts/diabimmune/base_lines_mlp.py

echo "All MLP baseline runs completed at: $(date)" | tee -a "$LOG_FILE"

