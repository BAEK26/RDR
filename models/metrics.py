from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import pairwise_distances
import numpy as np


def euc_dist(features, indices1, indices2, n_jobs=1):
    if isinstance(indices1, (int,np.integer)):
        indices1 = [indices1]
    if isinstance(indices2, (int,np.integer)):
        indices2 = [indices2]
    return pairwise_distances(features[indices1], features[indices2], metric='euclidean', n_jobs=n_jobs).squeeze()

def cos_dist(features, indices1, indices2, n_jobs=1):
    if isinstance(indices1, (int,np.integer)):
        indices1 = [indices1]
    if isinstance(indices2, (int,np.integer)):
        indices2 = [indices2]
    return pairwise_distances(features[indices1], features[indices2], metric='cosine', n_jobs=n_jobs).squeeze()

def config_dist(configurations, indices1, indices2, n_jobs=1):
    if isinstance(indices1, (int,np.integer)):
        indices1 = [indices1]
    if isinstance(indices2, (int,np.integer)):
        indices2 = [indices2]
    return pairwise_distances(configurations[indices1], configurations[indices2], metric='manhattan', n_jobs=n_jobs).squeeze().astype(int)