import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class RDR():
    def __init__(self, neighbors, configurations):
        super().__init__()
        self.similars = neighbors
        self.configs = configurations
    
    def selection(self, k=8, t=10):
        semantic_conf = self.configs[self.similars[:k]]
        prob = np.mean(semantic_conf, axis=0)
        C = np.where((prob==0) + (prob==1))[0]
        
        neg_configurations = np.mean(self.configs, axis=0)
        
        act_score = np.abs(prob - neg_configurations)
        Principal_Configuration = C[np.argsort(act_score[C])[::-1]][:t]

        val_PC = prob[Principal_Configuration]>0.5
        rdr_neurons = Principal_Configuration
        rdr_states = val_PC

        '''finding underlying_samples'''
        underlying_samples = []

        for i in range(self.configs.shape[0]):
            if np.sum(np.abs(self.configs[i][rdr_neurons]-np.array(rdr_states))) <= 0:
                underlying_samples.append(i)

        print(len(rdr_neurons),'decision boundaries are used for constructing relaxed decision region.')        
        print('The number of samples included in the Relaxed Decision Region:',len(underlying_samples))
        
        return underlying_samples, rdr_neurons, rdr_states
    
def visualize(sample_indices, images):
    rand_idx = np.random.choice(len(sample_indices), 10, replace=False)

    plt.figure(figsize=(9.5, 4))
    gs = gridspec.GridSpec(2, 5)
    gs.update(wspace=0, hspace=0)

    for kk in range(10):
        idx = sample_indices[rand_idx[kk]]
        img = images[idx].permute(1,2,0)
        ax = plt.subplot(gs[kk])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()