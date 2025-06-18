import json, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from data.nlp.helpers import char_to_token   # same as before
from pathlib import Path
import os

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def add_offsets(row):
    sent, ent = row["sentence"], row["entity"]
    try:
        start = sent.index(ent)          # because you wrote the sentence
    except ValueError:
        raise ValueError(f"Entity '{ent}' not found in sentence: {sent}")
    end   = start + len(ent)
    idx   = char_to_token(sent, (start, end))   # reuse helper
    row.update(char_span=(start, end),
               entity_token_idx=idx)
    return row


class HalluData(Dataset):
    data_path = Path(__file__).parent / "hallucination.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    def __init__(self, path=data_path):
        with open(path) as f:
            rows = [add_offsets(json.loads(l)) for l in f]
        self.rows = rows
    def __len__(self):  return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def get_split(_):   # keep API same as other datasets
    return HalluData()

if __name__ == "__main__":
    # for debugging
    dataset = get_split()
    print(f"Loaded {len(dataset)} samples.")
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}: {sample['sentence']}")
        print(f"Entity: {sample['entity']} at {sample['char_span']} (token idx: {sample['entity_token_idx']})")