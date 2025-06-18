"""
End-to-end RDR run over entity tokens in WikiNeural (English).

Usage
-----
python -m scripts.run_rdr_llama_ent --layer 27 --k 8 --t 10 --split test_en
python -m scripts.run_rdr_llama_ent --layer 27 --k 8 --t 10 --split test_en
"""

import argparse
import numpy as np

from models.language.llama_rdr          import LlamaRDR
from models.language.config_utils       import getconfigs_entities, config_dist
from models.rdr                         import RDR
from utils.visualize_text           import visualize_entity
from data.nlp import wikineural_ent, hallu


def main(args):
    # 0  Load dataset & model
    if args.dataset == "wikineural_ent":
        dataset = wikineural_ent.get_split(args.split)
    elif args.dataset == "hallucination":
        dataset = hallu.get_split(args.split)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    model   = LlamaRDR(target_layer=args.layer, capture_seq_pos=None)  # keep full seq

    # 1  Pick a random target instance
    rand_target = np.random.randint(len(dataset))
    rand_target = 59  # for debugging
    print(f"Target sample index: {rand_target}  | entity = {dataset[rand_target]['entity']}")

    # 2  Gather matrices
    feats, configs = getconfigs_entities(dataset, model,
                                         batch_size=args.batch_size)

    # 3  Distances
    dists      = config_dist(configs, rand_target)
    neighbours = np.argsort(dists)

    # 4  RDR
    rdr   = RDR(neighbours, configs)
    samp, neurons, states = rdr.selection(k=args.k, t=args.t)

    # 5  Visualise
    visualize_entity(samp, dataset, print_max=100)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=27)
    p.add_argument("--k",     type=int, default=8,  help="nearest neighbours")
    p.add_argument("--t",     type=int, default=10, help="principal neurons")
    p.add_argument("--split", type=str, default="test_en")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dataset", type=str, default="wikineural_ent",
                   choices=["wikineural_ent", "hallucination"])
    args = p.parse_args()

    main(args)
