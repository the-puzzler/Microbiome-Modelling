import csv
import os
import sys

# Ensure project root is on sys.path so absolute imports like `scripts.utils` work when running subfolder scripts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Paths
GINGIVITIS_TABLE = 'data/gingivitis/gingiva.csv'


def load_gingivitis_run_data(
    gingivitis_path=GINGIVITIS_TABLE,
    microbeatlas_path=None,
):
    """
    Read gingivitis SRA run IDs and map them to MicrobeAtlas SRS IDs.

    Returns:
        run_ids: set of SRA Run IDs found in gingivitis table
        SRA_to_micro: dict mapping SRR/ERR run ID -> MicrobeAtlas SRS
    """
    run_ids = set()
    with open(gingivitis_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get('Run', '').strip()
            if run_id:
                run_ids.add(run_id)

    # Map using headers from the mapped file (handles SRR/ERR/ERS etc.)
    acc_to_srs = shared_utils.build_accession_to_srs_from_mapped(shared_utils.MAPPED_PATH)
    SRA_to_micro = {rid: acc_to_srs[rid] for rid in run_ids if rid in acc_to_srs}

    mapped = len(SRA_to_micro)
    print(f'gingivitis runs: {len(run_ids)}, mapped to microbeatlas (via mapped headers): {mapped}')
    return run_ids, SRA_to_micro


def collect_micro_to_otus(SRA_to_micro, mapped_path=shared_utils.MAPPED_PATH):
    """
    Build SRS -> OTUs using the mapped MicrobeAtlas file only (no BIOM fallback).
    """
    needed_srs = set(SRA_to_micro.values())
    return shared_utils.collect_micro_to_otus_mapped(needed_srs, mapped_path)


def load_microbiome_model(checkpoint_path=shared_utils.CHECKPOINT_PATH):
    return shared_utils.load_microbiome_model(checkpoint_path)


def preview_prokbert_embeddings(prokbert_path=shared_utils.PROKBERT_PATH, limit=10):
    return shared_utils.preview_prokbert_embeddings(prokbert_path, limit)


def build_sample_embeddings(
    micro_to_otus,
    model,
    device,
    prokbert_path=shared_utils.PROKBERT_PATH,
    txt_emb=shared_utils.TXT_EMB,
):
    rename_map = None
    resolver = None
    try:
        if os.path.exists(shared_utils.RENAME_MAP_PATH):
            rename_map = shared_utils.load_otu_rename_map(shared_utils.RENAME_MAP_PATH)
            # Gingiva is bacterial; prefer B keys when both A/B exist
            resolver = shared_utils.build_otu_key_resolver(micro_to_otus, rename_map, shared_utils.PROKBERT_PATH, prefer='B')
    except Exception:
        rename_map = None
        resolver = None
    return shared_utils.build_sample_embeddings(
        micro_to_otus,
        model,
        device,
        prokbert_path=prokbert_path,
        txt_emb=txt_emb,
        rename_map=rename_map,
        resolver=resolver,
    )


def plot_dropout_summary(
    logits_t1,
    labels_dropout,
    title_prefix="Dropout prediction",
    xlim=None,
    ylim=None,
    save_path=None,
):
    """
    Create a 1x2 subplot figure:
    - Left: density hist of T1 logits, colored by dropped-out (1) vs persistent (0)
    - Right: ROC curve predicting dropout using -logit as the decision score

    Args:
        logits_t1 (array-like): raw logits at T1 for each OTU example
        labels_dropout (array-like): 1 if OTU dropped out at T2, else 0
        title_prefix (str): optional title prefix
    """
    import numpy as np

    logits_t1 = np.asarray(logits_t1, dtype=float)
    y_drop = np.asarray(labels_dropout, dtype=int)
    mask0 = y_drop == 0
    mask1 = y_drop == 1

    # Compute ROC using dropout decision score = -logit
    scores_drop = -logits_t1
    fpr, tpr, _ = roc_curve(y_drop, scores_drop)
    auc_val = roc_auc_score(y_drop, scores_drop) if np.unique(y_drop).size > 1 else float('nan')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Density histograms of logits
    ax = axes[0]
    if np.any(mask1):
        ax.hist(logits_t1[mask1], bins=40, density=True, alpha=0.5, label='Dropped out (1)')
    if np.any(mask0):
        ax.hist(logits_t1[mask0], bins=40, density=True, alpha=0.5, label='Persistent (0)')
    ax.set_xlabel('T1 logit')
    ax.set_ylabel('Density')
    ax.set_title(f'{title_prefix}: Logit densities')
    ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # ROC curve
    ax = axes[1]
    ax.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title_prefix}: ROC (dropout)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return fig, axes


def plot_colonisation_summary(
    logits_t1,
    labels_colonized,
    title_prefix="Colonisation prediction",
    save_path=None,
):
    """
    Similar to plot_dropout_summary but for colonization (gain):
    - Left: density of T1 logits for colonized vs non-colonized
    - Right: ROC curve using sigmoid(logit) as the score for colonization
    """
    import numpy as np

    logits_t1 = np.asarray(logits_t1, dtype=float)
    y = np.asarray(labels_colonized, dtype=int)
    mask1 = y == 1
    mask0 = y == 0

    probs = 1 / (1 + np.exp(-logits_t1))
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = roc_auc_score(y, probs) if np.unique(y).size > 1 else float('nan')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if np.any(mask1):
        ax.hist(logits_t1[mask1], bins=40, density=True, alpha=0.5, label='Colonized (1)')
    if np.any(mask0):
        ax.hist(logits_t1[mask0], bins=40, density=True, alpha=0.5, label='Did not colonize (0)')
    ax.set_xlabel('T1 logit (with target added)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title_prefix}: Logit densities')
    ax.legend()

    ax = axes[1]
    ax.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title_prefix}: ROC (colonisation)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return fig, axes
