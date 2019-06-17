# This file containes all the utility functions required for logistic regression
# The functions are based on the implementation of logistic regression using only numpy from another GitHub repo
# Link to GitHub repo [https://github.com/SamSamhuns/custom_linear_logistic_regr]
### m = number of training examples
### n = number of features/weights

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    ''' g(z) sigmoid '''
    return 1 / (1+np.exp(-z))

def y_pred(X_feat, theta_vector):
    ''' g(X.theta_vector)
    X_feat must have dimensions m*(n+1), y_target must have dimensions m*1
    theta_vector must have dimensions (n+1)*1'''
    return sigmoid(X_feat.dot(theta_vector))

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
        print("Error: dimensions of X_feat and y_target do not match", file=stderr)
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

def plot_boundary_line(theta_vector, X_feat, y_target):
    ''' Here x = feature x1 and y = feature x2
    Note: plt.show() is not called by default
    IMPORTANT: Only works when comparing the plots of two variables'''
    x_min, x_max, y_min, y_max = min(X_feat[:,1])-0.5, max(X_feat[:,1])+0.5, min(X_feat[:,2])-0.5, max(X_feat[:,2])+0.5 # find min,max of both x and y axis values

    x0, y0 = np.linspace(x_min, x_max, num=100), np.linspace(y_min, y_max, num=100) # gen evenly spaced 100 points between min, max values in the x and y axis
    xx, yy = np.meshgrid(x0, y0)
    '''
    np.ones((xx.ravel().shape[0],1)) generates a [(ravelled xx numpy arr len), 1] array which is concatenated with the np.c_[] to the ravelled xx and yy arrays
    Straightens the xx and yy arrays then concatenate them on axis=1 or rowise, X_all.shape = [xx.size * yy.size, 3]
    '''
    X_all = np.c_[np.ones((xx.ravel().shape[0],1)), xx.ravel(), yy.ravel()]  # for just the linear logistic regression part
    if theta_vector.size == 6: # 2nd degree polynomial logistic regression
        # engineering the new polynomial features in the form 1, a, b, a^2, b^2 and ab
        eng_feat1 = np.array(X_all[:,1:3]**2) # a^2 and b^2
        eng_feat2 = np.array(X_all[:,1]* X_all[:,2]).reshape(X_all.shape[0], 1) # ab
        X_all = np.c_[X_all, eng_feat1, eng_feat2] # np.concatenate((X_feat, eng_feat1, eng_feat2), axis=1)

    X_all_pred = y_pred(X_all, theta_vector)
    X_all_pred = np.array([1 if i >= 0.5 else 0 for i in X_all_pred])
    X_all_pred = X_all_pred.reshape(xx.shape)

    # plot the contours
    plt.contourf(xx, yy, X_all_pred, cmap=mpl.colors.ListedColormap(['#cc9880', '#7e97cc']) )

    plot_scatter(X_feat, y_target)
    plt.legend(loc='upper right')
    plt.title("Mapping binary variables")
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.grid(True)

def plot_loss_and_fit(X_feat, y_target, logistic_model):
    ''' function to print the loss curve and the boundary line
    IMPORTANT: Only works when comparing the plots of two variables'''
    plt.figure(figsize=[12,6])
    plt.subplot(1,2,1)
    plot_loss_curve(logistic_model.loss, logistic_model.epochs)
    plt.subplot(1,2,2)
    plot_boundary_line(logistic_model.theta_vector, X_feat, y_target)
    plt.tight_layout()
    plt.show()

def logistic_accuracy(predicted_labels, actual_labels):
    ''' function to calculate the accuracy of my model'''
    count = 0
    for i in range(len(actual_labels)):
        if predicted_labels[i] == actual_labels[i]:
            count += 1
    return count/(len(actual_labels))

def print_accuracy_best_git_parameters(X_feat, y_target, regr_model):
    ''' function to print the accuracy and the parameters of the logistic model'''
    pred = y_pred(X_feat, regr_model.theta_vector)
    pred = [1 if x >= .5 else 0 for x in pred]
    parameters = regr_model.theta_vector.reshape(1,regr_model.theta_vector.shape[0])
    print(f"Parameterx for the model = {parameters}, \nloss = {np.asscalar(regr_model.loss[-1])}, \naccuracy = {logistic_accuracy(pred, y_target)}")
