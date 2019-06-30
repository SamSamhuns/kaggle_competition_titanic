from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
cimport numpy as np
cimport cython

# This file containes all the utility functions required for logistic regression
# The functions are based on the implementation of logistic regression using only numpy from another GitHub repo
# Link to GitHub repo [https://github.com/SamSamhuns/custom_linear_logistic_regr]
### m = number of training examples
### n = number of features/weights

def sigmoid(z):
    ''' g(z) sigmoid '''
    return 1 / (1+np.exp(-z))

def y_pred(X_feat, theta_vector):
    ''' g(X.theta_vector)
    X_feat must have dimensions m*(n+1), y_target must have dimensions m*1
    theta_vector must have dimensions (n+1)*1'''
    return sigmoid(np.dot(X_feat, theta_vector))

def mean_norm(data):
    ''' mean normalization
    data must have dimensions m*(n+1)'''
    mean = np.mean(data[:,1:], axis=0)
    std = np.std(data[:,1:], axis=0)
    return (data[:,1:] - mean)/std

def logistic_loss(X_feat, y_target, theta_vector):
    ''' function to calculate the loss for logistic regression
    IMPORTANT: Extremely important to normalize data before calculating logistic loss'''
    if X_feat.shape[0] != y_target.shape[0]:
        print("Error: dimensions of X_feat and y_target do not match")
        return
    m_num = X_feat.shape[0]
    h_func = y_pred(X_feat, theta_vector)
    return (-1/m_num)*((np.transpose(y_target)).dot(np.log(h_func)) + (np.transpose(1-y_target)).dot(np.log(1-h_func)))

def plot_loss_curve(loss, iterations, log_mode=True):
    '''Note: plt.show() is not called by default in this function'''
    loss = [np.asscalar(i) for i in loss]
    if log_mode:
        plt.semilogx(range(iterations), loss)
    else:
        plt.plot(loss)
    plt.title("Loss overtime")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def logistic_accuracy(predicted_labels, actual_labels):
    ''' function to calculate the accuracy of my model'''
    cdef int count = 0
    cdef int actual_labels_len = len(actual_labels)
    for i in range(actual_labels_len):
        if predicted_labels[i] == actual_labels[i]:
            count += 1
    return count/(len(actual_labels))

def print_accuracy_best_git_parameters(X_feat, y_target, regr_model):
    ''' function to print the accuracy and the parameters of the logistic model'''
    pred = y_pred(X_feat, regr_model.theta_vector)
    pred = [1 if x >= .5 else 0 for x in pred]
    parameters = regr_model.theta_vector.reshape(1,regr_model.theta_vector.shape[0])
    print(f"Parameterx for the model = {parameters}, \nloss = {np.asscalar(regr_model.loss[-1])}, \naccuracy = {logistic_accuracy(pred, y_target)}")
