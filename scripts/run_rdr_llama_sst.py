# scripts/run_rdr_llama.py
import numpy as np
from models.language.llama_rdr import LlamaRDR
from models.language.config_utils import getconfigs, config_dist
from models.rdr import RDR
from utils.visualize_text import visualize_text
from data.nlp import sst2

train_data = sst2.get_split("train")     # returns torch Dataset
model = LlamaRDR(target_layer=27, capture_seq_pos=-1)

# 1️⃣ pick a target instance
rand_target = np.random.randint(len(train_data))
tar_label   = train_data[rand_target]["label"]

# 2️⃣ gather feature/config matrices
feats, configs = getconfigs(train_data, model)

# 3️⃣ compute distances
config_values   = config_dist(configs, rand_target, np.arange(configs.shape[1]))
config_similars = np.argsort(config_values)

# 4️⃣ build RDR
rdr = RDR(config_similars, configs)
rdr_samples, rdr_neurons, rdr_states = rdr.selection(k=8, t=10)

visualize_text(rdr_samples, train_data)
