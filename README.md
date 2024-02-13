# Relaxed Decision Region

Understanding Distributed Representations of Concepts in Deep Neural Networks without Supervision  
Wonjoon Chang* · Dahee Kwon* · Jaesik Choi (* Equal Contribution)  
\[[Paper](https://arxiv.org/abs/2312.17285)\]\[[Project]()\]  

This is the official pytorch implementation of Understanding Distributed Representations of Concepts in Deep Neural Networks without Supervision which is published on AAAI 2024. 

## Abstract
Understanding intermediate representations of the concepts learned by deep learning classifiers is indispensable for interpreting general model behaviors. In this paper, we propose a novel unsupervised method for discovering distributed representations of concepts by selecting a principal subset of neurons. Our empirical findings demonstrate that instances with similar neuron activation states tend to share coherent concepts. Based on the observations, the proposed method selects principal neurons that construct an interpretable region, namely a Relaxed Decision Region (RDR), encompassing instances with coherent concepts in the feature space. Our method identifies unlabeled data subclasses, misclassification causes, and revealing distinct representations across layers for deeper insights into deep learning mechanisms.

![image](./imgs/concept-img-rdr.png)

## Example
```python
import numpy as np
from models.configurations import *
from models.rdr import RDR, visualize

train_data = ''
model = ''

'''choosing target instance'''
rand_target = np.random.choice(len(train_data),1)[0]
tar_label = int(train_data[:][1][rand_target].numpy())
org_class = int(train_data.true_labels[rand_target])

print('Pred: ', class_dict[tar_label])
print('True: ', class_dict[org_class])

'''computing configuration distance'''
_, configs = getconfigs(train_data,model,tar_layer=27)
config_values = config_dist(configs, rand_target,np.arange(len(features)), n_jobs=4)
config_similars = np.argsort(config_values)

'''forming RDR'''
rdr = RDR(config_similars, configs)
rdr_samples, rdr_neurons, rdr_states = rdr.selection(k=8, t=10)

visualize(rdr_samples, train_data[:][0])
```
Please refer to the [notebook](./Relaxed-Decision-Region.ipynb)

## Citation
If you find this repo useful, please cite our paper:
```
@article{chang2023understanding,
  title={Understanding Distributed Representations of Concepts in Deep Neural Networks without Supervision},
  author={Chang, Wonjoon and Kwon, Dahee and Choi, Jaesik},
  journal={arXiv preprint arXiv:2312.17285},
  year={2023}
}
```
