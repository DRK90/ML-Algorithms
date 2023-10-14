import numpy as num
import pandas as pd

def calculateRoot(dataframe):
    def entropy(column):
        classType, counts = num.unique(column, return_counts=True)
        entropy = -num.sum([(counts[i]/num.sum(counts))*num.log2(counts[i]/num.sum(counts)) for i in range(len(classType))])
        return entropy

    def informationGain(testFeature, target="class"):
        #calculate initial entropy
        totalEntropy = entropy(dataframe[target])

        #after the split what would the entropy be?
        vals, counts = num.unique(dataframe[testFeature], return_counts=True)
        weightedEntropy = num.sum([(counts[i]/num.sum(counts)) * entropy(dataframe.where(dataframe[testFeature]==vals[i]).dropna()[target]) for i in range(len(vals))])

        #find the gain
        infoGain = totalEntropy - weightedEntropy
        return infoGain
    
    def intrinsicValue(testFeature):
        vals, counts = num.unique(dataframe[testFeature], return_counts=True)
        iv = num.sum([-(counts[i]/len(dataframe)) * num.log2(counts[i]/len(dataframe)) for i in range(len(vals))])
        return iv
    
    bestRatio = float('-inf')
    bestFeature = None
    bestValue = None
    allFeatureGainRatios = []

    for feature in dataframe.columns.drop('class'):
        #check on each unique value
        uniqueValues = dataframe[feature].unique()
        bestRatioFeature = float('-inf')
        bestValueFeature = None
        for val in uniqueValues:
            dataBelow = dataframe[dataframe[feature] <= val]
            dataAbove = dataframe[dataframe[feature] > val]

            gainOnThisSplit = informationGain(feature, "class")

            iv = intrinsicValue(feature)

            gainRatio = gainOnThisSplit / iv if iv !=0 else 0

            if gainRatio > bestRatioFeature:
                bestRatioFeature = gainRatio
                bestValueFeature = val

            if gainRatio > bestRatio:
                bestRatio = gainRatio
                bestFeature = feature
                bestValue = val
        allFeatureGainRatios.append({'gainRatio': bestRatioFeature,
                                    'feature': feature,
                                    'value': bestValueFeature
                                    })
    allFeatureGainRatios.append({
        'gainRatio:': bestRatio,
        'bestFeature': bestFeature,
        'bestValue': bestValue
    })

    return allFeatureGainRatios

