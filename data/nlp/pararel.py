# pararel_data.py
import torch, json
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from data.nlp.helpers import char_to_token          # unchanged
from pathlib import Path
from tqdm import tqdm
# same tokenizer you already use elsewhere
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


def add_offsets(row):
    """
    Enrich a raw Pararel row with:
      • sentence         – instantiated template containing subject & object
      • entity           – alias for object
      • char_span        – (start, end) in characters
      • entity_token_idx – first sub-token idx of entity in that sentence
      • prediction / correct flags (mirrors your HalluData layout)
    """
    subj   = row["subject"]
    ent    = row["entity"] # row.get("object")
    rel_tmpl = row.get("relation") # row.get("template")

    # 1) Try the existing query; 2) fall back to full template; 3) append object.
    base_sent = row.get("query") or row.get("sentence") or ""
    if ent not in base_sent:                     # usual case for Pararel
        if rel_tmpl:                             # e.g. “[X] is located in [Y].”
            sent = rel_tmpl.replace("[X]", subj).replace("[Y]", ent)
        else:                                    # ultimate fallback
            sent = f"{base_sent.strip()} {ent}"
    else:
        sent = base_sent

    # char offsets & token index
    start = sent.index(ent)           # guaranteed to exist now
    end   = start + len(ent)
    tok_idx = char_to_token(sent, (start, end))+1  # your helper

    row.update(
        sentence          = sent,
        entity            = ent, # ground-truth object
        char_span         = (start, end),
        entity_token_idx  = tok_idx,
        prediction        = row["prediction"],       
        correct           = row["correct"]       
    )
    return row

def filter_entity(row):
    """
    Only use entity
    """
    row["sentence"] = " "+row.get("entity", "")
    row = add_offsets(row)  # reuse the same offset logic
    decoded_entity = tok.decode(tok.encode(row['sentence'])[row['entity_token_idx']])
    if decoded_entity.strip() != row["entity"].strip():
        # raise ValueError(f"Entity token mismatch: {row['sentence']} vs {row['entity']}")
        # print(f"Entity token mismatch: #{row['entity']}# vs #{decoded_entity}#")
        pass
    row['decoded_entity'] = decoded_entity
    return row
    

class PararelData(Dataset):
    """Pararel → RDR-ready dataset, same interface as HalluData."""
    def __init__(self, split="train", entity_option=None):
        # raw = load_dataset("coastalcph/pararel_patterns", split=split)  # 

        with open('/data8/baek/dehallu/RDR/data/nlp/pararel_llama31_predictions.jsonl', 'r') as f:
            raw = [json.loads(line) for line in f.readlines()]
        # enrich & drop rows where token mapping failed (idx == 0)
        if "entity only" in entity_option.lower():
            self.rows = [
            ex for ex in tqdm(map(filter_entity, raw), total = len(raw), desc= "Add offsets", unit="rows") if ex["entity"] == ex["decoded_entity"].strip()
        ]
        else:
            self.rows = [
                ex for ex in tqdm(map(add_offsets, raw), total = len(raw), desc= "Add offsets", unit="rows") if ex["entity_token_idx"] != 0
            ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def get_split(split="train", entity_option=None):   # keeps the external API identical
    return PararelData(split=split, entity_option=entity_option)


# quick sanity check
if __name__ == "__main__":
    ds = get_split()
    print(f"Loaded {len(ds)} Pararel samples with entity spans.")
    for i in range(3):
        ex = ds[i]
        print(f"▶ {ex['sentence']}")
        print(f"   entity='{ex['entity']}' char_span={ex['char_span']} token_idx={ex['entity_token_idx']}")
        print(f"   entity token: #{tok.decode(tok.encode(ex['sentence'])[ex['entity_token_idx']])}#")
