#!/usr/bin/env python3
"""
Compare the text-input projection weights between:
  1) A checkpoint believed to be "no-text" trained:
       data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt
  2) A text-trained checkpoint:
       data/model/checkpoint_epoch_0_final_newblack_2epoch.pt
  3) A freshly initialised MicrobiomeTransformer (random init)

We report:
  - Basic stats (mean, std, min, max) of input_projection_type2 weights
  - Frobenius norms
  - Cosine similarity between flattened weight matrices:
        fresh vs no-text
        fresh vs text-trained
        no-text vs text-trained

If the "no-text" checkpoint really never saw text, its input_projection_type2
should be very close to the fresh random init, and quite different from the
text-trained checkpoint.
"""

import os
import sys

import torch
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import utils as shared_utils  # noqa: E402
from model import MicrobiomeTransformer  # noqa: E402


CKPT_NO_TEXT = "data/model/checkpoint_epoch_0_final_newblack_2epoch_notextabl.pt"
CKPT_WITH_TEXT = "data/model/checkpoint_epoch_0_final_newblack_2epoch.pt"


def flatten_weight(mat: torch.Tensor) -> np.ndarray:
    return mat.detach().cpu().numpy().reshape(-1).astype(np.float64)


def describe_weights(name: str, w: torch.Tensor):
    arr = flatten_weight(w)
    print(f"\n[{name}] input_projection_type2 statistics:")
    print(f"  shape: {tuple(w.shape)}")
    print(f"  mean : {arr.mean(): .6f}")
    print(f"  std  : {arr.std(): .6f}")
    print(f"  min  : {arr.min(): .6f}")
    print(f"  max  : {arr.max(): .6f}")
    print(f"  ||W||_F: {np.linalg.norm(arr): .6f}")
    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main():
    # Fresh random model with same dims as utils
    fresh_model = MicrobiomeTransformer(
        input_dim_type1=shared_utils.OTU_EMB,
        input_dim_type2=shared_utils.TXT_EMB,
        d_model=shared_utils.D_MODEL,
        nhead=shared_utils.NHEAD,
        num_layers=shared_utils.NUM_LAYERS,
        dim_feedforward=shared_utils.DIM_FF,
        dropout=shared_utils.DROPOUT,
    )

    print("Loading no-text checkpoint:", CKPT_NO_TEXT)
    model_no_text, _ = shared_utils.load_microbiome_model(CKPT_NO_TEXT)

    print("Loading text-trained checkpoint:", CKPT_WITH_TEXT)
    model_with_text, _ = shared_utils.load_microbiome_model(CKPT_WITH_TEXT)

    w_fresh = fresh_model.input_projection_type2.weight
    w_no = model_no_text.input_projection_type2.weight
    w_with = model_with_text.input_projection_type2.weight

    a_fresh = describe_weights("fresh", w_fresh)
    a_no = describe_weights("no-text", w_no)
    a_with = describe_weights("text-trained", w_with)

    print("\nCosine similarities between flattened weight matrices:")
    print(f"  fresh vs no-text     : {cosine_similarity(a_fresh, a_no): .6f}")
    print(f"  fresh vs text-trained: {cosine_similarity(a_fresh, a_with): .6f}")
    print(f"  no-text vs text-trained: {cosine_similarity(a_no, a_with): .6f}")


if __name__ == "__main__":
    main()

