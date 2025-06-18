"""
Grid-sweep of RDR hyper-parameters.

Example
-------
python -m scripts.rdr_sweep \
       --layers 20,24,27,30 \
       --t_vals 5,8,10,12 \
       --k_vals 4,8,12 \
       --split test_en \
       --outfile rdr_sweep.log
"""

import argparse, time, json, sys, os
from pathlib import Path

import numpy as np
from models.language.llama_rdr          import LlamaRDR
from models.language.config_utils       import getconfigs_entities, config_dist
from models.rdr                         import RDR
from data.nlp.wikineural_ent            import get_split
import torch

# ------------------------------------------------------------------ helpers
GREEN, RESET = "\x1b[32m", "\x1b[0m"
def cgreen(msg): return f"{GREEN}{msg}{RESET}"

def log_append(path: Path, entry: dict):
    """Append a JSON line so it’s easy to parse later."""
    with path.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ main
def main(args):
    layers   = [int(x) for x in args.layers.split(",")]
    t_vals   = [int(x) for x in args.t_vals.split(",")]
    k_vals   = [int(x) for x in args.k_vals.split(",")]
    dataset  = get_split(args.split)
    outpath  = Path(args.outfile)

    model = LlamaRDR(target_layer=0, capture_seq_pos=None)
    configs_by_layer = {}

    for L in layers:
        print(cgreen(f"\n=== Layer {L} ==="))
        model.tlayer = L
        if L not in configs_by_layer:
            _, configs = getconfigs_entities(dataset, model,
                                            batch_size=args.batch_size)
            configs_by_layer[L] = configs
        else:
            configs = configs_by_layer[L]

        for k in k_vals:
            for t in t_vals:
                anchor_idx = np.random.randint(len(dataset))
                anchor_idx = 3  # for debugging
                dists = config_dist(configs, anchor_idx)
                nn_order = np.argsort(dists)

                rdr = RDR(nn_order, configs)
                samples, neurons, states = rdr.selection(k=k, t=t)

                entry = {
                    "time":        time.strftime("%Y-%m-%d %H:%M:%S"),
                    "layer":       L,
                    "k":           k,
                    "t":           t,
                    "anchor_idx":  int(anchor_idx),
                    "anchor_ent":  dataset[anchor_idx]["entity"],
                    "region_size": int(len(samples)),
                    "sample_indices": [int(s) for s in samples[:5]],
                    "sample_entities": [dataset[s]["entity"] for s in samples[:5]],
                }
                log_append(outpath, entry)

                rs = entry["region_size"]
                print(f" L={L:2d}  k={k:2d}  t={t:2d}  |  size={rs:4d}  anchor='{entry['anchor_ent']}'")
    print(cgreen(f"\nSweep done. Results saved to {outpath.resolve()}"))


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layers",  default="27", help="comma list, e.g. 20,24,27")
    p.add_argument("--t_vals",  default="5,8,10", help="comma list of t values")
    p.add_argument("--k_vals",  default="4,8,16", help="comma list of k values")
    p.add_argument("--split",   default="test_en")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--outfile", default="rdr_sweep.log")
    args = p.parse_args()

    main(args)