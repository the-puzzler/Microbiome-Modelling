#!/usr/bin/env bash
# Run main zero-shot evaluations + IBS + infants env prediction from repo root
# Usage: bash run_main_evals.sh

set -u  # treat unset variables as an error

# Where to store all stdout/stderr
LOG_FILE="experiment_log_$(date +%Y%m%d_%H%M%S).txt"

# Use non-interactive backend so matplotlib doesn't try to open windows
export MPLBACKEND=Agg

echo "Writing all outputs to: $LOG_FILE"
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

# Gingivitis zero-shot dropout / colonisation
run_step "Gingivitis zero-shot dropout" \
  uv run python -u scripts/gingivitis/dropout_test.py

run_step "Gingivitis zero-shot colonisation" \
  uv run python -u scripts/gingivitis/colonisation_test.py

# DIABIMMUNE zero-shot dropout / colonisation
run_step "DIABIMMUNE zero-shot dropout" \
  uv run python -u scripts/diabimmune/dropout_test.py

run_step "DIABIMMUNE zero-shot colonisation" \
  uv run python -u scripts/diabimmune/colonisation_test.py

# Snowmelt zero-shot dropout / colonisation
run_step "Snowmelt zero-shot dropout" \
  uv run python -u scripts/snowmelt/dropout_test.py

run_step "Snowmelt zero-shot colonisation" \
  uv run python -u scripts/snowmelt/colonisation_test.py

# IBS prediction (cross-country)
run_step "IBS cross-country prediction (frozen embeddings + logistic regression)" \
  uv run python -u scripts/IBS/predict_ibs.py

# Infants environment prediction
run_step "Infants environment prediction (frozen embeddings + logistic regression)" \
  uv run python -u scripts/infants/predict_env.py

echo "All runs completed at: $(date)" | tee -a "$LOG_FILE"
