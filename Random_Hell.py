from __future__ import print_function
import numpy as np
import sys
import time
from Util import *
from Random_CLT import CLT


class RANDOM_HELL():

    def __init__(self):
        self.n_components = 0  # number of components
        self.tree_probs = []  # mixture probabilities
        self.clt_list = []  # List of Tree Bayesian networks

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''

    def normalize(self, array):
        return array / np.sum(array)

    def learn(self, dataset, k, r):
        for i in range(k):
            self.tree_probs.append(1/k)
            tree = CLT()
            bootstrap_indices = np.random.choice(np.arange(dataset.shape[0]), size=dataset.shape[0], replace=True)
            bootstrap = dataset[bootstrap_indices]
            tree.learn(bootstrap, r)
            self.clt_list.append(tree)




    """
        Compute the log-likelihood score of the dataset
    """

    def computeLL(self, dataset):
        ll = 0.0
        for i in range(dataset.shape[0]):
            summer = 0
            for k in range(len(self.clt_list)):
                summer += self.tree_probs[k] * self.clt_list[k].getProb(dataset[i])
            ll += np.log(summer)
        return ll / dataset.shape[0]


'''
    After you implement the functions learn and computeLL, you can learn a mixture of trees using
    To learn Chow-Liu trees, you can use
    mix_clt=MIXTURE_CLT()
    ncomponents=10 #number of components
    max_iter=50 #max number of iterations for EM
    epsilon=1e-1 #converge if the difference in the log-likelihods between two iterations is smaller 1e-1
    dataset=Util.load_dataset(path-of-the-file)
    mix_clt.learn(dataset,ncomponents,max_iter,epsilon)

    To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    mix_clt.computeLL(dataset)/dataset.shape[0]
'''





