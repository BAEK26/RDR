from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



def getconfigs(train_data, model, tar_layer):

    instances = train_data[:][0][0]
    feat, config = model.instance2featconfig(instances.cuda(), tar_layer)

    feat_dim = feat.flatten().shape[0]
    config_dim = config.flatten().shape[0]

    n_sample = len(train_data)

    feats = np.empty((n_sample, feat_dim))
    configs = np.empty((n_sample, config_dim))

    data_loader = DataLoader(train_data, batch_size=64, shuffle=False)

    i=0
    bs=64

    for instances,_ in tqdm(data_loader):

        model.eval()
        with torch.no_grad():
            feat, config = model.instance2featconfig(instances.cuda(), tar_layer)
        feat = feat.cpu().numpy().reshape(len(instances),-1)
        config = config.cpu().numpy().reshape(len(instances),-1)
        
        start = i*bs
        end = start + len(instances)
        
        feats[start:end] = feat
        configs[start:end] = config
        
        i += 1
    return feats, configs