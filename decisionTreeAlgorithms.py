import numpy as num
import pandas as pd

class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, data=None):
        self.feature = feature  # feature split on
        self.value = value      # value to split on
        self.left = left        # left child
        self.right = right      # right
        self.data = data 

def calculateRoot(dataframe):
    """
    find the root of the remaining data (recursive)
    Parameters: dataframe, target
    returns: root
    
    """
    def entropy(column):
        classType, counts = num.unique(column, return_counts=True)
        entropy = -num.sum([(counts[i]/num.sum(counts))*num.log2(counts[i]/num.sum(counts)) for i in range(len(classType))])
        return entropy

    def informationGain(testFeature, testValue, target="class"):
        #calculate initial entropy
        totalEntropy = entropy(dataframe[target])

        dataBelow = dataframe[dataframe[testFeature] <= testValue]
        dataAbove = dataframe[dataframe[testFeature] > testValue]
        
        probBelow = len(dataBelow) / len(dataframe)
        probAbove = len(dataAbove) / len(dataframe)
        
        weightedEntropy = (probBelow * entropy(dataBelow[target])) + (probAbove * entropy(dataAbove[target]))

        infoGain = totalEntropy - weightedEntropy
        return infoGain
    
    def intrinsicValue(testFeature, testValue):
        dataBelow = dataframe[dataframe[testFeature] <= testValue]
        dataAbove = dataframe[dataframe[testFeature] > testValue]
        
        probBelow = len(dataBelow) / len(dataframe)
        probAbove = len(dataAbove) / len(dataframe)
        
        if probBelow == 0 or probAbove == 0:
            return 0

        iv = -(probBelow * num.log2(probBelow) + probAbove * num.log2(probAbove))
        return iv


            
    
    bestRatio = float('-inf')
    bestFeature = None
    bestValue = None
    allFeatureGainRatios = []

    for feature in dataframe.columns.drop('class'):
        totalEntropy = entropy(dataframe['class'])
        #base case, when all of the class is the same
        if totalEntropy == 0:
            return Node(data=int(dataframe["class"].iloc[0]))
        
        #check on each unique value
        uniqueValues = dataframe[feature].unique()
        for val in uniqueValues:

            gainOnThisSplit = informationGain(feature, val, "class")

            iv = intrinsicValue(feature, val)

            gainRatio = gainOnThisSplit / iv if iv !=0 else 0


            if gainRatio > bestRatio:
                bestRatio = gainRatio
                bestFeature = feature
                bestValue = val
        #allFeatureGainRatios.append({'gainRatio': bestRatioFeature,
         #                           'feature': feature,
          #                          'value': bestValueFeature
           #                         })
    #define the current node
    currentNode = Node(feature=bestFeature, value=bestValue)

    #split the data for recursion
    trainingDataLeft = dataframe[dataframe[bestFeature] <= bestValue]
    trainingDataRight = dataframe[dataframe[bestFeature] > bestValue]

    #recurse
    currentNode.left = calculateRoot(trainingDataLeft)
    currentNode.right = calculateRoot(trainingDataRight)

    allFeatureGainRatios.append({
        'gainRatio': bestRatio,
        'bestFeature': bestFeature,
        'bestValue': bestValue
    })

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
        if prediction == row["class"]:
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

    classValues, counts = num.unique(validationData["class"], return_counts=True)
    node.data = classValues[num.argmax(counts)]
    
    # get the accuracy of the post pruned tree
    
    postAccuracy = prunePredict(nodeCopy, validationData)
    
    # If pruning doesn't increase the accuracy, restore the children
    if postAccuracy <= preAccuracy:
        node.left = tempLeft
        node.right = tempRight
        node.data = None
    else:
        print("test")
    
    




