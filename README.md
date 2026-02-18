# Microbiome Model

This repo contains a transformer-based microbiome representation model plus evaluation scripts for multiple downstream tasks.

## Install with `uv`

```bash
uv sync
```

Run scripts with:

```bash
uv run python <script_path.py>
```

## Main evaluation scripts

- `run_main_evals.sh`: runs the core zero-shot evaluations and prediction tasks.
- `run_mlp_baselines.sh`: runs MLP baselines for key datasets.

From repo root:

```bash
bash run_main_evals.sh
bash run_mlp_baselines.sh
```

## Model location

- Architecture: `model.py`
- Shared loading/config paths: `scripts/utils.py`
- Default checkpoint path referenced by scripts: `data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt`

## Data location

- Local data root: `data/`
- Temporary download link (replace with real URL): `https://TEMP-DATA-LINK.example.com`

## What the model does

The model embeds sets of microbiome OTUs into a latent representation and supports downstream tasks including:

- Zero-shot dropout prediction
- Zero-shot colonisation prediction
- IBS cross-country prediction
- Infant environment prediction
- Comparative rollout trajectory analysis/visualization

## DIABIMMUNE one-out-one-in anchored visualization

Script:

- `scripts/rollout/plot_diabimmune_oneoutoneinanchored_trajectory_overlay.py`

Run:

```bash
uv run python scripts/rollout/plot_diabimmune_oneoutoneinanchored_trajectory_overlay.py
```

Output figure:

- `data/rollout_metropolis/diabimmune_rollout_trajectory_overlay_oneoutoneinanchored.png`
