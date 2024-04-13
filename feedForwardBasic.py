import numpy as num
import pandas as pd
import matplotlib.pyplot as plt
import dataFunctions as funcs

def train(df, target):
    """
    Trains feedforward network with backpropagation
    Parameters: 
    df(dataFrame): training data
    target(string): classification target column
    Returns: 
    trainedWeights(list): weights to use for future tests
    """
    #first encode the classes to be 0 and 1 to represent binary classification.  2->0 4->1
    df[target] = df[target].map({2:0, 4:1})
    #one hot encode target so the shape will match the output layer
    outputColumns = pd.get_dummies(df[target], prefix=target)
    df = pd.concat([df, outputColumns], axis=1)
    #capture the target values 
    y = df[outputColumns.columns].values
    #drop target to capture feature values
    features = df.drop([target] + list(outputColumns.columns), axis=1)
    #get the matrix of values
    X = features.values

    #sigmoid function to be used in calculating probability
    def sigmoid(x):
        return 1/(1 + num.exp(-x))
    
    def softmax(x):
        #exponentiate each value of x after subtracting the max from the set
        xExponent = num.exp(x - num.max(x))
        #normalize the values so they sum to 1 across the row
        return xExponent / xExponent.sum(axis=1, keepdims=True)
    
    def calculateLoss(actual, predicted):
       #calculate cross entropy loss
        return -num.sum(actual * num.log(predicted)) / actual.shape[0]
    
    #initialize parameters
    def initParams(inputSize, hiddenSize1, hiddenSize2, outputSize):
        #create random weights for each layer connecting one layer to the next, here is input layer to first hidden layer
        w1 = num.random.randn(inputSize, hiddenSize1)
        #set inital bias to 0 for each layer starting with the first hidden layer
        b1 = num.zeros((1, hiddenSize1))
        w2 = num.random.randn(hiddenSize1, hiddenSize2)
        b2 = num.zeros((1, hiddenSize2))
        w3 = num.random.randn(hiddenSize2, outputSize)
        b3 = num.zeros((1, outputSize))
        return w1, b1, w2, b2, w3, b3
    
    #forward pass
    def feedForward(X, w1, b1, w2, b2, w3, b3):
        #get the weighted sum of inputs with the assigned weights and biases
        weightedSum1 = num.dot(X, w1) + b1
        #apply sigmoid function on the weighted sum to get the output
        outputHidden1 = sigmoid(weightedSum1)
        weightedSum2 = num.dot(outputHidden1, w2) + b2
        outputHidden2 = sigmoid(weightedSum2)
        weightedSumFinal = num.dot(outputHidden2, w3) + b3
        #apply the softmax function to get the final output
        outputFinal = softmax(weightedSumFinal)
        return outputHidden1, outputHidden2, outputFinal

    
    #backward pass
    def backProp(X, Y, outputHidden1, outputHidden2, outputFinal, W2, W3):
        numEntries = X.shape[0]
        #find differenfe between output of final layer and true values
        differenceOutputFinalAndActual = outputFinal - Y
        gradientLoss3 = num.dot(outputHidden2.T, differenceOutputFinalAndActual) / numEntries
        gradientBias3 = num.sum(differenceOutputFinalAndActual, axis=0, keepdims=True) / numEntries
        differenceOutput2andFinal = num.dot(differenceOutputFinalAndActual, W3.T) * outputHidden2 * (1 - outputHidden2)
        gradientLoss2 = num.dot(outputHidden1.T, differenceOutput2andFinal) / numEntries
        gradientBias2 = num.sum(differenceOutput2andFinal, axis=0, keepdims=True) / numEntries
        differenceOutput1and2 = num.dot(differenceOutput2andFinal, W2.T) * outputHidden1 * (1 - outputHidden1)
        gradientLoss1 = num.dot(X.T, differenceOutput1and2) / numEntries
        gradientBias1 = num.sum(differenceOutput1and2, axis=0, keepdims=True) / numEntries
        return gradientLoss1, gradientBias1, gradientLoss2, gradientBias2, gradientLoss3, gradientBias3
    
    #update parameters with the calculated weights and biases gained from backprop
    def updateParams(W1, b1, W2, b2, W3, b3, gradientWeight1, gradientBias1, gradientWeight2, gradientBias2, gradientWeight3, gradientBias3, learningRate):
        W1 -= learningRate * gradientWeight1
        b1 -= learningRate * gradientBias1
        W2 -= learningRate * gradientWeight2
        b2 -= learningRate * gradientBias2
        W3 -= learningRate * gradientWeight3
        b3 -= learningRate * gradientBias3
        return W1, b1, W2, b2, W3, b3
    
    #count number of features in the df
    inputSize = features.shape[1]
    #define the number of neurons to be equal to the midpoint of the input and output size
    hiddenSize1 = (features.shape[1] + len(num.unique(y)))//2
    hiddenSize2 = (features.shape[1] + len(num.unique(y)))//2
    #number of classes
    outputSize = len(num.unique(y))
    learningRate = .01
    iterations = 1000

    #get initial weights and biases
    w1, b1, w2, b2, w3, b3 = initParams(inputSize, hiddenSize1, hiddenSize2, outputSize)
    lossOverTime = []
    for i in range(iterations):
        #get the outputs from each pf the layers
        outputHidden1, outputHidden2, outputFinal = feedForward(X, w1, b1, w2, b2, w3, b3)
        #calculate loss each time to see improvements
        loss = calculateLoss(y, outputFinal)
        gradientWeight1, gradientBias1, gradientWeight2, gradientBias2, gradientWeight3, gradientBias3 = backProp(X, y, outputHidden1, outputHidden2, outputFinal, w2, w3)
        w1, b1, w2, b2, w3, b3 = updateParams(
            w1, b1, w2, b2, w3, b3, 
            gradientWeight1, gradientBias1, gradientWeight2, gradientBias2, gradientWeight3, gradientBias3, 
            learningRate)
        lossOverTime.append(loss)    

    #set to 1 to show plot
    if 1==1:
        plt.plot(lossOverTime)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss over time')
        plt.savefig('lossovertime.pdf', format='pdf') 

        plt.show()
    return w1, b1, w2, b2, w3, b3

def test(df, target, _w1, _b1, _w2, _b2, _w3, _b3):
    """
    tests feedforward network with backpropagation
    Parameters: 
    df(dataFrame): training data
    target(string): classification target column
    Returns: 
    trainedWeights(list): weights to use for future tests
    """
    #first encode the classes to be 0 and 1 to represent binary classification.  2->0 4->1
    df[target] = df[target].map({2:0, 4:1})
    #one hot encode target so the shape will match the output layer
    outputColumns = pd.get_dummies(df[target], prefix=target)
    df = pd.concat([df, outputColumns], axis=1)
    #capture the target values 
    y = df[outputColumns.columns].values
    #drop target to capture feature values
    features = df.drop([target] + list(outputColumns.columns), axis=1)
    #get the matrix of values
    X = features.values

    #sigmoid function to be used in calculating probability
    def sigmoid(x):
        return 1/(1 + num.exp(-x))
    
    def softmax(x):
        #exponentiate each value of x after subtracting the max from the set
        xExponent = num.exp(x - num.max(x))
        #normalize the values so they sum to 1 across the row
        return xExponent / xExponent.sum(axis=1, keepdims=True)
    
    def calculateLoss(actual, predicted):
       #calculate cross entropy loss
        return -num.sum(actual * num.log(predicted)) / actual.shape[0]
    
    #initialize parameters
    def initParams(inputSize, hiddenSize1, hiddenSize2, outputSize):
        #create random weights for each layer connecting one layer to the next, here is input layer to first hidden layer
        w1 = num.random.randn(inputSize, hiddenSize1)
        #set inital bias to 0 for each layer starting with the first hidden layer
        b1 = num.zeros((1, hiddenSize1))
        w2 = num.random.randn(hiddenSize1, hiddenSize2)
        b2 = num.zeros((1, hiddenSize2))
        w3 = num.random.randn(hiddenSize2, outputSize)
        b3 = num.zeros((1, outputSize))
        return w1, b1, w2, b2, w3, b3
    
    #forward pass
    def feedForward(X, w1, b1, w2, b2, w3, b3):
        #get the weighted sum of inputs with the assigned weights and biases
        weightedSum1 = num.dot(X, w1) + b1
        #apply sigmoid function on the weighted sum to get the output
        outputHidden1 = sigmoid(weightedSum1)
        weightedSum2 = num.dot(outputHidden1, w2) + b2
        outputHidden2 = sigmoid(weightedSum2)
        weightedSumFinal = num.dot(outputHidden2, w3) + b3
        #apply the softmax function to get the final output
        outputFinal = softmax(weightedSumFinal)
        return outputHidden1, outputHidden2, outputFinal

    
    #backward pass
    def backProp(X, Y, outputHidden1, outputHidden2, outputFinal, W2, W3):
        numEntries = X.shape[0]
        #find differenfe between output of final layer and true values
        differenceOutputFinalAndActual = outputFinal - Y
        gradientLoss3 = num.dot(outputHidden2.T, differenceOutputFinalAndActual) / numEntries
        gradientBias3 = num.sum(differenceOutputFinalAndActual, axis=0, keepdims=True) / numEntries
        differenceOutput2andFinal = num.dot(differenceOutputFinalAndActual, W3.T) * outputHidden2 * (1 - outputHidden2)
        gradientLoss2 = num.dot(outputHidden1.T, differenceOutput2andFinal) / numEntries
        gradientBias2 = num.sum(differenceOutput2andFinal, axis=0, keepdims=True) / numEntries
        differenceOutput1and2 = num.dot(differenceOutput2andFinal, W2.T) * outputHidden1 * (1 - outputHidden1)
        gradientLoss1 = num.dot(X.T, differenceOutput1and2) / numEntries
        gradientBias1 = num.sum(differenceOutput1and2, axis=0, keepdims=True) / numEntries
        return gradientLoss1, gradientBias1, gradientLoss2, gradientBias2, gradientLoss3, gradientBias3
    
    #update parameters with the calculated weights and biases gained from backprop
    def updateParams(W1, b1, W2, b2, W3, b3, gradientWeight1, gradientBias1, gradientWeight2, gradientBias2, gradientWeight3, gradientBias3, learningRate):
        W1 -= learningRate * gradientWeight1
        b1 -= learningRate * gradientBias1
        W2 -= learningRate * gradientWeight2
        b2 -= learningRate * gradientBias2
        W3 -= learningRate * gradientWeight3
        b3 -= learningRate * gradientBias3
        return W1, b1, W2, b2, W3, b3
    
    #count number of or rows in the df
    inputSize = features.shape[1]
    #define the number of neurons to be equal to the midpoint of the input and output size
    hiddenSize1 = (features.shape[1] + len(num.unique(y)))//2
    hiddenSize2 = (features.shape[1] + len(num.unique(y)))//2
    #number of classes
    outputSize = len(num.unique(y))
    learningRate = .01
    iterations = 1000

    #get trained weights
    w1, b1, w2, b2, w3, b3 = _w1, _b1, _w2, _b2, _w3, _b3
    outputHidden1, outputHidden2, outputFinal = feedForward(X, w1, b1, w2, b2, w3, b3)
    predicted = num.argmax(outputFinal, axis=1)
    actual = num.argmax(y, axis=1)
    accuracy = num.mean(predicted == actual)
    return accuracy

    

