import json
import matplotlib.pyplot as plt
import numpy as np
import pylatexenc
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.model_selection import train_test_split
from models import get_QCNN, PCA_DR, UMAP_DR, AE_DR
from data import get_dataloader
import argparse
from utils import callback_graph
from utils import eval


algorithm_globals.random_seed = 12345


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument("--DR", type=str, default="PCA"
                        ,help="What dimentionality reduction method should be used? [UMAP, PCA, AE]")
    parser.add_argument("--iter", type=int, default=200
                        ,help="Number of iterations (defult is 200)")   
    parser.add_argument("--data_size", type=int, default=250
                        ,help="Number of data samples (defult is 250)")
    parser.add_argument("--AE_epoch", type=int, default=20
                        ,help="Number of epoch in AE (defult is 20)")                       
    args = parser.parse_args()

    print("-"*10)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-"*10)
    


    qnn = get_QCNN(args)

    dataloader = get_dataloader(args)
    testloader = get_dataloader(args, train=False)

    if args.DR.lower() == 'pca':    
        pca_8d, proj_8d, targets = PCA_DR(args, dataloader)
        pca_8d, proj_8d_test, targets_test = PCA_DR(args, testloader, pca_8d)
    elif args.DR.lower() == 'umap':
        umap_8d, proj_8d, targets = UMAP_DR(args, dataloader)
        umap_8d, proj_8d_test, targets_test = UMAP_DR(args, testloader, umap_8d)
    elif args.DR.lower() == 'ae':
        AE_8d, proj_8d, targets = AE_DR(args, dataloader)
        AE_8d, proj_8d_test, targets_test = AE_DR(args, testloader, AE_8d)



    classifier = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=args.iter), callback=callback_graph)


    x = np.asarray(proj_8d)
    y = np.asarray(targets)
    x_test = np.asarray(proj_8d_test)
    y_test = np.asarray(targets_test)

    classifier.fit(x, y)



    print(eval(classifier,x,y))

    print(eval(classifier,x_test,y_test))

    
