
from __future__ import print_function
import numpy as np
import sys
import time
from Util import *
from CLT_class import CLT

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks




    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def normalize(self, array):
        return array / np.sum(array)

    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        # For each component and each data point, we have a weight
        weights = np.zeros((n_components, dataset.shape[0])) # gamma^i_ks?
        self.mixture_probs = np.random.random(n_components)
        summ = np.sum(self.mixture_probs)
        self.mixture_probs = self.mixture_probs/summ

        for k in range(n_components):
            self.clt_list.append(CLT())
            self.clt_list[k].learn(dataset)

        ll_prime = -np.inf
        for itr in range(max_iter):
            for k in range(n_components):
                for i, sample in enumerate(dataset):
                    weights[k][i] = self.mixture_probs[k]*self.clt_list[k].getProb(sample)
            weights = weights/weights.sum(axis=1)[:,None]



    # M-step: Update the Chow-Liu Trees and the mixture probabilities
            for k in range(n_components):
                self.clt_list[k].update(dataset, weights[k, :])
    # Your code for M-Step here
            ll = self.computeLL(dataset)
            self.mixture_probs = np.sum(weights, axis=1)
            self.mixture_probs = self.mixture_probs / dataset.shape[0]
            if abs(ll - ll_prime) < epsilon:
                break
            ll_prime = ll


    
    """
        Compute the log-likelihood score of the dataset
    """
    def computeLL(self, dataset):
        ll = 0.0
        for i in range(dataset.shape[0]):
            summer = 0
            for k in range(len(self.clt_list)):
                summer += self.mixture_probs[k] * self.clt_list[k].getProb(dataset[i])
            ll += np.log(summer)
        return ll/dataset.shape[0]
    
