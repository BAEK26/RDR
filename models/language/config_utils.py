"""
Utility helpers for Relaxed-Decision-Region on language models.

Functions
---------
getconfigs(dataset, rdr_model, *, batch_size=8, device='cuda', pin_memory=False)
    → (feats, configs)   # both np.ndarray

config_dist(configs, target_idx, feature_idx=None)
    → np.ndarray[float]  # distance of every row to the target row
"""

from typing import Sequence, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import pairwise_distances  # ★ NEW import


# ---------------------------------------------------------------------
# Distance  (new version using sklearn)
# ---------------------------------------------------------------------

def config_dist(
    configs: np.ndarray,
    target_idx: int,
    feature_idx: Sequence[int] | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Hamming distance between every row and configs[target_idx],
    implemented via sklearn.pairwise_distances (metric='manhattan').

    Parameters
    ----------
    configs : (N, D) ndarray of {0,1}
    target_idx : int
    feature_idx : 1-D index array or None
    n_jobs : int
        Parallel jobs for sklearn.

    Returns
    -------
    dists : (N,) ndarray of int
        Hamming distance for each sample (0-D inclusive).
    """
    # 선택된 feature만 쓰고 싶으면 슬라이스
    if feature_idx is not None:
        cfg  = configs[feature_idx,:]
    else:
        cfg  = configs
    tgt  = [configs[target_idx]]                # (1, D)

    # 맨해튼(=L1)-거리 → 이진 벡터에서는 곧 해밍 거리
    dists = pairwise_distances(cfg, tgt, metric="manhattan", n_jobs=n_jobs)

    # (N,1) → (N,)  &  정수 캐스팅
    return dists.ravel().astype(np.int32)



# ---------------------------------------------------------------------
# Feature / configuration extractor
# ---------------------------------------------------------------------
def _collate_prompts(batch):
    """Default collate: just keep dicts as a list."""
    return batch


def getconfigs(
    dataset,
    rdr_model,
    *,
    batch_size: int = 8,
    device: str = "cuda",
    pin_memory: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hidden features and binary activation configurations for **all**
    samples in a Dataset.

    Each feature/config corresponds to the single token position chosen when
    the `LlamaRDR` wrapper was instantiated (`capture_seq_pos`).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Must return a dict with at least a `"prompt"` field.
    rdr_model : models.language.llama_rdr.LlamaRDR
    batch_size : int
    device : str
    pin_memory : bool

    Returns
    -------
    feats   : (N, hidden_size) np.ndarray
    configs : (N, hidden_size) np.ndarray of {0,1}
    """
    rdr_model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_prompts,
        pin_memory=pin_memory,
    )

    feat_chunks, conf_chunks = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting features and configs"):
            # batch is a list of dicts -> list[str]
            prompts = [b["prompt"] for b in batch]
            toks = rdr_model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            feat, conf = rdr_model.instance2featconfig(toks["input_ids"])
            # feat/conf: [bs, hidden_size]
            feat_chunks.append(feat.float().cpu().numpy())
            conf_chunks.append(conf.int().cpu().numpy())

    feats   = np.concatenate(feat_chunks, axis=0)
    configs = np.concatenate(conf_chunks, axis=0)
    return feats, configs

def getconfigs_entities(dataset, rdr_model, batch_size=8):
    feats, configs = [], []
    def _collate_entities(batch):
        return batch
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_entities,)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting features and configs"):
            ids = rdr_model.tokenize([b["sentence"] for b in batch]).to("cuda")
            pos = torch.tensor([b["entity_token_idx"] for b in batch],
                               dtype=torch.long, device="cuda")
            # print(pos)
            f, c = rdr_model.instance2featconfig_pos(ids["input_ids"], pos)
            feats.append(f.cpu().numpy())
            configs.append(c.cpu().numpy())
    return np.vstack(feats), np.vstack(configs)
