import numpy as num
import pandas as pd
import matplotlib.pyplot as plt

def train(df, target):
    """
    Trains logistic regression model
    Parameters: 
    df(dataFrame): training data
    target(string): classification target column
    Returns: 
    trainedWeights(list): weights to use for future tests
    """
    #first encode the classes to be 0 and 1 to represent binary classification.  2->0 4->1
    df[target] = df[target].map({2:0, 4:1})
    #capture the target values 
    y = df[target].values
    #drop target to capture feature values
    features = df.drop(target, axis=1)
    #get the matrix of values
    X = features.values
    #add the bias term
    X = num.insert(X, 0, 1, axis=1)

    #sigmoid function to be used in calculating probability
    def sigmoid(x):
        return 1/(1 + num.exp(-x))
    
    def cost(X, y, weights):
        #calculate the predicted probability for each row in the dataframe.  numpy dot will perform matrix multiplcation on each row with the weights
        predictedProbability = sigmoid(num.dot(X, weights))
        #calculate the cross entropy loss
        cost = -y * num.log(predictedProbability) - (1-y) * num.log(1-predictedProbability)
        return cost.mean()
    
    #gradient descent
    def gradientDescent(X, y, weights, learning_rate, iterations):
        costOverTime = []
        for i in range(iterations):
            predictions = sigmoid(num.dot(X, weights))
            #matrix multiplication of the transpose of X (each feature as a row), with the each error
            gradient = num.dot(X.T, (predictions - y)) / y.size
            #subtract from the weights the calculated step, try to minimize loss each time
            weights -= learning_rate * gradient
            #add the cost for visualization
            costOverTime.append(cost(X,y,weights))
        return weights, costOverTime
    
    #set the weight for each feature to initially be 0
    initWeights = num.zeros(X.shape[1])
    learningRate = .01
    iterations = 1000
    trainedWeights, costOverTime = gradientDescent(X, y, initWeights, learningRate, iterations)
    #set to 1 to show plot
    if 1==1:
        plt.plot(costOverTime)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost over time')
        plt.savefig('costOverTime.pdf', format='pdf') 
        plt.show()
    return trainedWeights

def test(df, target, weights):
    """
    test logistic regression model
    Parameters: 
    df(dataFrame): training data
    target(string): classification target column
    weights(numpy array): trained weights from training
    Returns: 
    accuracy(int): how accurate the test was
    """
    #first encode the classes to be 0 and 1 to represent binary classification.  2->0 4->1
    df[target] = df[target].map({2:0, 4:1})
    #capture the target values 
    y = df[target].values
    #drop target to capture feature values
    features = df.drop(target, axis=1)
    #get the matrix of values
    X = features.values
    #add the bias term
    X = num.insert(X, 0, 1, axis=1)

    #sigmoid function to be used in calculating probability
    def sigmoid(x):
        return 1/(1 + num.exp(-x))
    
   

    predictions = sigmoid(num.dot(X, weights))
    #get the result of each value in the array if its >=0.5, we say its 1 (or 4), else 0 (or 2)
    predictions = (predictions >= 0.5).astype(int)
    accuracy = (predictions == y).mean()
    return accuracy
    

