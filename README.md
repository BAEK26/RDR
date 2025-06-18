# RDR for LLMs: Discovering Token Decision Boundaries in Large Language Models

This repository extends the Relaxed Decision Region (RDR, Neuron Configuration Distance) framework to Large Language Models (LLMs) such as Llama 3.1. Our goal is to identify and interpret the decision boundaries of tokens by analyzing distributed neuron activations, providing insight into how LLMs represent and distinguish concepts at the token level.

![RDR Concept](./imgs/concept-img-rdr.png)

## Overview

RDR is an unsupervised method for discovering distributed representations of concepts by selecting a principal subset of neurons. By analyzing neuron activation patterns, RDR finds interpretable regions—**Relaxed Decision Regions**—that correspond to coherent token-level concepts in LLMs. This approach helps reveal how LLMs make token-level decisions and where their boundaries lie.

## Key Features

- **LLM Support:** Directly supports Llama 3.1 and can be adapted to other transformer-based models.
- **Token-level Analysis:** Finds decision boundaries for tokens, not just for images or classes.
- **Unsupervised:** No labeled data required for discovering neuron configurations and naming each region.
- **Visualization:** Tools for visualizing decision regions and inspecting representative samples.

## Example Usage

Below is a minimal example of running RDR on Llama 3.1 to find token decision boundaries:

```python
import numpy as np
from models.language.llama_rdr import LlamaRDR
from models.configurations import getconfigs
from models.metrics import config_dist
from models.rdr import RDR
from utils.visualize_text import visualize_text

# Load your dataset (must yield dicts with 'prompt' and 'true_labels' or 'label')
train_data = ...  # e.g., a torch Dataset of tokenized prompts

# Initialize LlamaRDR for a specific layer
model = LlamaRDR(target_layer=27, capture_seq_pos=-1)  # -1 = last token or entity token

# Extract neuron configurations for all samples
feats, configs = getconfigs(train_data, model, tar_layer=27)

# Choose a target instance
rand_target = np.random.choice(len(train_data), 1)[0]

# Compute configuration distances to all other samples
config_values = config_dist(configs, rand_target, np.arange(len(configs)), n_jobs=4)
config_similars = np.argsort(config_values)

# Form the Relaxed Decision Region (RDR)
rdr = RDR(config_similars, configs)
rdr_samples, rdr_neurons, rdr_states = rdr.selection(k=8, t=10)

# Visualize representative samples in the RDR
visualize_text(rdr_samples, train_data)
```

For a full pipeline, see [`scripts/run_rdr_llama_ent.py`](./scripts/run_rdr_llama_ent.py) and the [notebook](./Relaxed-Decision-Region.ipynb)(for original RDR).

## Reference

If you use this codebase, please cite the original RDR paper:

```
@article{chang2023understanding,
  title={Understanding Distributed Representations of Concepts in Deep Neural Networks without Supervision},
  author={Chang, Wonjoon and Kwon, Dahee and Choi, Jaesik},
  journal={arXiv preprint arXiv:2312.17285},
  year={2023}
}
```
<!-- 
```
@misc{baek2025rdrllm,
  author = {Jongeun Baek},
  title = {RDR for LLMs: Discovering Token Decision Boundaries in Large Language Models},
  year = {2025},
  howpublished = {\\url{https://github.com/BAEK26/RDR}}
}
``` -->
