import os.path
from MIXTURE_CLT import MIXTURE_CLT
from Random_CLT import CLT
from CLT_class import CLT
from Util import *
from Random_Hell import RANDOM_HELL

def run_chow_lu(filenames):
    for file in filenames:
        tree = CLT()
        file1 = file + ".ts.data"
        filename = os.path.join("dataset", file1)
        dataset = Util.load_dataset(filename)
        tree.learn(dataset)
        file1 = file + ".test.data"
        test_set = os.path.join("dataset", file1)
        test_set = Util.load_dataset(test_set)
        ll = tree.computeLL(test_set)/test_set.shape[0]
        print("The log-likelihood on the test set for the Chow-Lu Tree is: {}".format(ll))

def run_mixture_trees(filenames):

    ks = [2, 5, 10, 20]
    for file in filenames:
        best_ks = []
        file1 = file + ".ts.data"
        filename = os.path.join("dataset", file1)
        dataset = Util.load_dataset(filename)

        for k in ks:
            tree = MIXTURE_CLT()
            tree.learn(dataset, n_components=k)
            validset = file + ".valid.data"
            valid = os.path.join("dataset", validset)
            validset = Util.load_dataset(valid)
            ll = tree.computeLL(validset)
            print("For the dataset: {} Using the EM Mixture Model with k={} the loglikelihood of the validation set is {}".format(file, k, ll))

def run_random_hell(filenames):
    ks = [2, 5, 7, 10]
    rs = [1, 5, 10, 20, 25]
    for file in filenames:
        file1 = file + ".ts.data"
        filename = os.path.join("dataset", file1)
        dataset = Util.load_dataset(filename)
        for k in ks:
            for r in rs:
                tree = RANDOM_HELL()
                tree.learn(dataset, k=k, r=r)
                validset = file + ".valid.data"
                valid = os.path.join("dataset", validset)
                validset = Util.load_dataset(valid)
                ll = tree.computeLL(validset)
                print("For the Dataset: {}  Using a Random Forest with k={} and r={}, the loglikelihood is: {}".format(file, k, r, ll))

if __name__ == '__main__':
    filenames = ["accidents", "baudio", "bnetflix", "jester", "kdd", "msnbc", "nltcs", "plants", "pumsb_star", "tretail"]
    #run_chow_lu(filenames)
    run_mixture_trees(filenames)
    #run_random_hell(filenames)
