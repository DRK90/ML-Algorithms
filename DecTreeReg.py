import numpy as num
import pandas as pd

class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, data=None):
        self.feature = feature  # feature split on
        self.value = value      # value to split on
        self.left = left        # left child
        self.right = right      # right
        self.data = data 

def calculateRoot(dataframe, target="Rings"):
    """
    find the root of the remaining data (recursive)
    Parameters: dataframe, target
    returns: root
    
    """

    def mse(column):
        meanValue = num.mean(column)
        return num.mean((column - meanValue)**2)
    
    def getSplitError(feature, val):
        dataBelow = dataframe[dataframe[feature] <= val]
        dataAbove = dataframe[dataframe[feature] > val]

        probBelow = len(dataBelow) / len(dataframe)
        probAbove = len(dataAbove) / len(dataframe)

        weightedMse = (probBelow * mse(dataBelow[target])) + (probAbove * mse(dataAbove[target]))
        return weightedMse
    
    def getSplits(dataframe, feature):
        percentiles = [10,40,70,90]
        potentialSplits = dataframe[feature].quantile(q=[p/100 for p in percentiles]).unique()
        return potentialSplits
        
    bestError = float('inf')
    bestFeature = None
    bestValue = None

    for feature in dataframe.columns.drop(target):
        totalVariance = mse(dataframe[target])
        #base case, when all of the class is the same
        if totalVariance == 0:
            return Node(data=num.mean(dataframe[target]))
        
        #check on each unique value for sex (still call them splits)
        if feature == 'Sex':
            splits = dataframe[feature].unique()
        else:
            splits = getSplits(dataframe, feature)

        for val in splits:

            currentError = getSplitError(feature, val)
            if currentError < bestError:
                bestError = currentError
                bestFeature = feature
                bestValue = val
    
    #return leaf, if theres no reduction in error
    if bestFeature is None:
        return Node(data=num.mean(dataframe[target]))
    
    currentNode = Node(feature=bestFeature, value=bestValue)

    #split data for recursive call
    trainingDataLeft = dataframe[dataframe[bestFeature] <= bestValue]
    trainingDataRight = dataframe[dataframe[bestFeature] > bestValue]

    #recurse
    currentNode.left = calculateRoot(trainingDataLeft, target=target)
    currentNode.right = calculateRoot(trainingDataRight, target=target)

    return currentNode

def printTree(node, indent=""):
    if not node:
        return

    # Leaf node, print the class
    if node.data is not None:
        print(f"{indent}Leaf: Class={node.data}")
        return

    # not leaf, print the feature and value to split
    print(f"{indent}Split on {node.feature} <= {node.value}")
    print(f"{indent}Left:")
    printTree(node.left, indent + "  ")

    print(f"{indent}Right:")
    printTree(node.right, indent + "  ")

def predict(instance, node):
    
    # return the class when we reach the leaf
    if node.data is not None:
        return node.data
    
    # go left or right depending on the value we split on
    if instance[node.feature] <= node.value:
        return predict(instance, node.left)
    else:
        return predict(instance, node.right)
    
def prunePredict(node, validateData):
    correct = 0
    for index, row in validateData.iterrows():
        prediction = predict(row, node)
        if prediction == row["Rings"]:
            correct += 1
    return correct / len(validateData)

def getApplicableData(targetNode, df, rootNode):
    if rootNode == targetNode:
        return df
    
    if rootNode.data is not None:
        return None
    
    leftData = df[df[rootNode.feature] <= rootNode.value]
    rightData = df[df[rootNode.feature] > rootNode.value]

    leftResult = getApplicableData(targetNode, leftData, rootNode.left)
    rightResult = getApplicableData(targetNode, rightData, rootNode.right)

    if leftResult is not None:
        return leftResult
    return rightResult
    
def prune(node, validationData, nodeCopy):

    # when we reach the leaf, return, base case
    if node.left is None and node.right is None:
        return    
    
    # prune left, then right
    if node.left:
        prune(node.left, validationData, nodeCopy)
    if node.right:
        prune(node.right, validationData, nodeCopy)
    
    preAccuracy = prunePredict(nodeCopy, validationData)
    
    # get the current left and right to restore once accuracy drops
    tempLeft = node.left
    tempRight = node.right
    
    # Temporarily turn the current node into a leaf node
    node.left = None
    node.right = None
    
    # prune by replacing the node with a leaf with the most common class value
    applicableData = getApplicableData(node, validationData, nodeCopy)

    classValues, counts = num.unique(validationData["Rings"], return_counts=True)
    node.data = classValues[num.argmax(counts)]
    
    # get the accuracy of the post pruned tree
    
    postAccuracy = prunePredict(nodeCopy, validationData)
    
    # If pruning doesn't increase the accuracy, restore the children
    if postAccuracy < preAccuracy:
        node.left = tempLeft
        node.right = tempRight
        node.data = None
    
    




