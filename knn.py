import numpy as np
import matplotlib.pyplot as plt
from run_knn import run_knn
from utils import load_train, load_valid, load_test


def knn():
    train_data, train_labels = load_train()

    #for validation
    valid_data, valid_labels = load_valid()
    
    #for test
    #valid_data, valid_labels = load_test()
    
    values = [1, 3, 5, 7, 9]
    ratio = []
    for k in values:
        c = 0
        prediction_labels = run_knn(k, train_data, train_labels, valid_data)
        
        for i in range(len(valid_labels)):
            if valid_labels[i] == prediction_labels[i]:
                c += 1
        ratio.append(float(c) / len(prediction_labels))

    plt.plot(values, ratio)
    
    #for validation
    plt.axis([1, 9, 0.81, 0.87])
    
    #for test
    #plt.axis([1, 9, 0.87, 0.95])
    
    plt.show()
