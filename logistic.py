""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function
    y = []
    b = weights[-1]
    w = weights[:-1]
    sig_list = []
    #print ("len of data",len(data))
    for i in xrange(len(data)):
        sig_val = sigmoid(np.dot(w.T, data[i]) + b)
        sig_list.append(sig_val)
    
    for i in sig_list:
        y.append(i)
    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    #ce
    ce = -1 * np.sum(targets * np.log(np.array(y).reshape(-1,1)) + (1 - targets) * np.log(1 - np.array(y).reshape(-1,1)))
    
    #frac_correct
    c = 0
    for i in xrange(len(targets)):
        if targets[i] == round(y[i]):
            c += 1
    frac_correct = float(c) / len(targets)
    
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function
    y = logistic_predict(weights, data)

    f = 0
    b = weights[-1]
    w = weights[:-1]
    sig_list = []
    df_list = []
    
    #f
    for i in xrange(len(targets)):
        sig_val = sigmoid(np.dot(w.T, data[i]) + b)
        sig_list.append(sig_val)
            
    sig_vector = np.array(sig_list).reshape(-1, 1) 
    f = -1 * np.sum(targets * np.log(sig_vector) + (1 - targets) * np.log(1 - sig_vector))  
    deriv_b = -1 * np.sum(targets - sig_list) 
    
    #dff
    dff = -1 * np.dot((targets - sig_vector).T, data)
    df = np.append(dff, deriv_b).reshape(-1, 1)  
    
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    # TODO: Finish this function
    
    f = 0
    b = weights[-1]
    w = weights[:-1]
    sig_list = []
    df_list = []
        
    #f
    for i in xrange(len(targets)):
        sig_val = sigmoid(np.dot(w.T, data[i]) + b)
        sig_list.append(sig_val)
                
    sig_vector = np.array(sig_list).reshape(-1, 1) 
    f = -1 * np.sum(targets * np.log(sig_vector) + (1 - targets) * np.log(1 - sig_vector))
    deriv_b = -1 * np.sum(targets - sig_list)
        
    #penalized logistic regression
    f = f + (hyperparameters/2) * np.dot(w.T, w)
            
    #dff
    dff = -1 * np.dot((targets - sig_vector).T, data) + hyperparameters * w.T
    df = np.append(dff, deriv_b).reshape(-1, 1)
        
    #y
    y = logistic_predict(weights, data)
        
        
    return f, df, y
