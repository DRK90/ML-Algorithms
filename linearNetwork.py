import numpy as num
import pandas as pd
import matplotlib.pyplot as plt

def train(df, target):
    """
    Trains linear network for regression data set
    Parameters: 
    df(dataFrame): training data
    target(string): classification target column
    Returns: 
    trainedWeights(list): weights to use for future tests
    """

    #capture the target values 
    y = df[target].values
    #drop target to capture feature values
    features = df.drop(target, axis=1)
    #get the matrix of values
    X = features.values
    #add the bias term
    X = num.insert(X, 0, 1, axis=1)

    #find the product of the matrix multiplication of the features with the weights
    def calculate(X, weights):
        return num.dot(X, weights)
    
    #find the mean square error
    def error(actual, predicted):
        return ((actual - predicted) ** 2).mean()
    
    #gradient descent
    def gradientDescent(X, y, lr=.01, i=100):
        #initialize weights to 0
        weights = num.zeros(X.shape[1])
        n = len(y)
        errorHistory=[]

        for x in range(i):
            #use the weights value each feature accordingly
            predicted = calculate(X, weights)
            #transpose of X to get each feature's significance and compare it against the different, then multiply by the learning rate to adjust the weights
            weights -= lr * (2/n) * num.dot(X.T, predicted - y)
            currentError = error(y, predicted)
            errorHistory.append(currentError)

        return weights, errorHistory
    
    weights, errorHistory = gradientDescent(X, y)
    plt.plot(errorHistory)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Error Over Iterations')
    plt.savefig('MSEpoint001.pdf', format='pdf') 
    plt.show()
    return weights

def test(df, target, weights):
    """
    test linear network for regression data set
    Parameters: 
    df(dataFrame): training data
    target(string): classification target column
    Returns: 
    accuracy(int): 
    """

    #capture the target values 
    y = df[target].values
    #drop target to capture feature values
    features = df.drop(target, axis=1)
    #get the matrix of values
    X = features.values
    #add the bias term
    X = num.insert(X, 0, 1, axis=1)

    #find the product of the matrix multiplication of the features with the weights
    def calculate(X, weights):
        return num.dot(X, weights)
    
    predictions = calculate(X, weights)
    mse = ((predictions - y)**2).mean()
    return mse
    