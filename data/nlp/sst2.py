"""
Tiny wrapper around GLUE-SST2 that matches the keys expected
by the vision-era RDR code (`prompt`, `label`, `true_labels`).
"""

from __future__ import annotations
from typing import Literal

from datasets import load_dataset
from torch.utils.data import Dataset


class SST2Dataset(Dataset):
    """
    Sentiment-classification split of GLUE’s SST-2.
    """
    def __init__(
        self,
        split: Literal["train", "validation", "test"] = "train",
    ):
        super().__init__()
        ds = load_dataset("glue", "sst2", split=split)
        self.texts   = ds["sentence"]
        self.labels  = ds["label"]          # 0 = negative, 1 = positive

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        lbl = int(self.labels[idx])
        return {
            "prompt": self.texts[idx],
            "label":  lbl,
            "true_labels": lbl,   # keep the old field-name for compatibility
        }


# ---------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------
def get_split(split: str = "train") -> SST2Dataset:
    """
    Shortcut mirroring `dataset.dataset` in the vision code.

    Examples
    --------
    >>> train_data = get_split("train")
    >>> sample = train_data[0]
    >>> sample.keys()
    dict_keys(['prompt', 'label', 'true_labels'])
    """
    return SST2Dataset(split=split)
