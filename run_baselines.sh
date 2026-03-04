#!/usr/bin/env bash
# Run all baseline scripts (traditional + MLP) for gingivitis, snowmelt, and DIABIMMUNE.
# Usage: bash run_baselines.sh

set -u  # treat unset variables as an error

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/baselines_log_$(date +%Y%m%d_%H%M%S).txt"

# Use non-interactive backend so matplotlib doesn't try to open windows
export MPLBACKEND=Agg

echo "Writing all baseline outputs to: $LOG_FILE"
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

# Gingivitis baselines
run_step "Gingivitis baseline colonisation (LogReg/RandForest)" \
  uv run python -u scripts/gingivitis/baselines_colonisation.py
run_step "Gingivitis baseline dropout (LogReg/RandForest)" \
  uv run python -u scripts/gingivitis/baselines_binary_taxa.py
run_step "Gingivitis MLP baselines" \
  uv run python -u scripts/gingivitis/base_lines_mlp.py

# Snowmelt baselines
run_step "Snowmelt baseline colonisation (LogReg/RandForest)" \
  uv run python -u scripts/snowmelt/baselines_colonisation.py
run_step "Snowmelt baseline dropout (LogReg/RandForest)" \
  uv run python -u scripts/snowmelt/baselines_binary_taxa.py
run_step "Snowmelt MLP baselines" \
  uv run python -u scripts/snowmelt/base_lines_mlp.py

# DIABIMMUNE baselines
run_step "DIABIMMUNE baseline colonisation (LogReg/RandForest)" \
  uv run python -u scripts/diabimmune/baselines_colonisation.py
run_step "DIABIMMUNE baseline dropout (LogReg/RandForest)" \
  uv run python -u scripts/diabimmune/baselines_binary_taxa.py
run_step "DIABIMMUNE MLP baselines" \
  uv run python -u scripts/diabimmune/base_lines_mlp.py

# Extract random-split + grouped-CV baseline summaries from this log
run_step "Extract baseline summaries from log" \
  uv run python -u scripts/checks/extract_baseline_results.py --log "$LOG_FILE"

echo "All baseline runs completed at: $(date)" | tee -a "$LOG_FILE"

