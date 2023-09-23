import pandas as pd
import numpy as np
import dataAlgorithms as da

def fillMissingWithMean(df):
    """
    Identify '?' or '' and replace with nan, then replace nan with the mean of the column.

    Parameters:
    df (dataframe): dataframe you want to replace values in

    Returns: a copy of the dataframe with the missing values replaced
    """
    #replace ?, '', with numpy's nan.  This allows us to use fillna to replace the value with a mean
    df.replace(['?',''], pd.NA, inplace = True)  
    #use an in-place replacement to convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
    #get mean for each column
    columnMeans = df.mean()    
    # fill missing values with mean
    df.fillna(columnMeans, inplace=True)
    return df

def replaceOrdinalslWithIntegers(df, ordinals, integers, column):
    """ 
    Replace ordinals (eg: education level) with numerically encoded values that correspond to the category

    Parameters:
    df (dataframe): dataframe you want to replace ordinals in
    ordinals (list): list of ordinals to be replaced, to be in order with the integers list
    integers (list): corresponding values to the ordinals
    column (string): name of the column in which the ordinals will be replaced

    Returns: a copy of the dataframe with the ordinals replaced
    """
    df[column].replace(ordinals, integers, inplace = True)
    return df

def oneHotEncodeColumn(df, column):
    """
    Use pandas get_dummies to replace a column with categorical data into a number of columns each with true/false if its the named value

    Parameters:
    df(dataframe): dataframe on which the encoding needs to take place
    column(string): name of the column that needs to be encoded

    Returns: a copy of the dataframe with the one hot encoding applied to the selected column    
    """
    return pd.get_dummies(df, columns = column, dtype=int)

def discretizeEqualWidth(df, column, numberOfBins):
    """
    Use pandas cut function to return discretized categorical data. pass which column to categorize, and the number of bins to put them in.  more options can be found...

    Parameters:
    df(dataframe): dataframe on which to discretize
    column(string): name of the column to discretize
    numberOfBins(int): specify number of bins to seperate the data into

    Returns: copy of the dataframe with the selected column discretized into equal width bins
    """
    df[column] = pd.cut(df[column], bins=numberOfBins)
    return df


def discretizeEqualFrequency(df, column, numberOfBins):
    """
    Use pandas qcut function to return data broken into buckets based on sample quantiles. numberOfBins here refers to the number of quantiles.

    Parameters:
    df(dataframe): dataframe on which to discretize
    column(string): name of the column to discretize
    numberOfBins(int): specify number of bins to seperate the data into
    
    Returns: copy of the dataframe with the selected column discretized into bins with equal frequency
    """
    df[column] = pd.qcut(df[column], q=numberOfBins)
    return df

def zScoreStandardize(df, column):
    """
    Find z score standardization by subtracting the mean value of the column from the column then dividing by the standard deviation

    Parameters:
    df(dataframe): dataframe to standardize
    column(string): name of the column within the dataframe to standardize

    Returns: copy of the dataframe with the selected column standardized
    """
    df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df

def hyperParameterTuning(combinedData, columns = [], regression=0):
    """
    Specify the columns to test and tune. manually enter the "test" that will be be used to tune

    Parameters:
    combinedData(list): Test Data, and training data broken into 2
    columns(list): list of attributes that are the candidate paramters
    regression(int): 1=yes.  will select mean rather than mode

    Returns: score of the paramters being checked
    """
    if regression == 0:
        trainingSet1Mode = combinedData[0]['class'].mode()
        trainingSet2Mode = combinedData[1]['class'].mode()
        testSetMode = combinedData[2]['class'].mode()
    else:
        trainingSet1Mode = combinedData[0]['area'].mean()
        trainingSet2Mode = combinedData[1]['area'].mean()
        testSetMode = combinedData[2]['area'].mean()

    combinedResults = [trainingSet1Mode, trainingSet2Mode, testSetMode]
    return combinedResults

def crossValidationKby2Classification(df, k=5):
    """
    Seperate data into 80% training and 20% test. Then, seperate the training data in half. All of these "buckets" are have an equal portion of class distribution.
    
    Parameters:
    df(dataframe): dataframe to cross validate
    k(int): defaulted to 5. Represents the number of times
    
    """
    #next break the dataframe in 2 one with each class value, this will assist in stratification 
    #THIS WILL NEED TO BE UPDATED IF MORE THAN 2 CLASSES
    testResult1 = []
    #collect the best returned parameter
    parameterCollector = pd.DataFrame(columns=['1','3','5','7'])
    #testCollector used for second part where we have the best parameter
    testCollector = pd.DataFrame(columns=['best parameter'])
    #initialize to collect best parameter after first part of experiement
    highestParameter = 0

    #EDIT KNN will be done first

    #First find the best parameters (Dont do this for edited KNN)
    if 1==0:
        for i in range(k):
            class1 = df[df['class']==4]
            class2 = df[df['class']==2]
            #class3 = df[df['class']==2]
            #class4 = df[df['class']==1]
            trainingData1 = class1.sample(frac=.8)
            trainingData2 = class2.sample(frac=.8)
            #trainingData3 = class3.sample(frac=.8)
            #trainingData4 = class4.sample(frac=.8)
            trainingData = pd.concat([trainingData1, trainingData2], axis=0)
            #trainingData = pd.concat([trainingData1, trainingData2, trainingData3, trainingData4], axis=0)
            testData1 = class1.drop(trainingData1.index)
            testData2 = class2.drop(trainingData2.index)
            #testData3 = class3.drop(trainingData3.index)
            #testData4 = class4.drop(trainingData4.index)
            #testData = pd.concat([testData1, testData2, testData3, testData4], axis = 0)        
            testData = pd.concat([testData1, testData2], axis = 0)    

            trainingDataSample1 = trainingData.sample(frac=.5, random_state = 1)
            trainingDataSample2 = trainingData.drop(trainingDataSample1.index)

            combinedData = [trainingDataSample1, trainingDataSample2, testData]

            #Collect the best parameter over the 5 experiements, this will be 10 sets        
            parameterCollector = pd.concat([parameterCollector, da.knnTest(combinedData)], axis=0)
        #print(parameterCollector)
        parameterAverages = parameterCollector.mean()
        highestParameter = parameterAverages.idxmax()
        highestParameter = int(highestParameter)
        #print(f'Column Averages:\n {parameterAverages} \n Highest Average: {highestParameter}')

    #Do 5 loops again, only with the parameter that was determined to be the best
    for i in range(k):
        class1 = df[df['class']==4]
        class2 = df[df['class']==2]
        #class3 = df[df['class']==2]
        #class4 = df[df['class']==1]
        trainingData1 = class1.sample(frac=.8)
        trainingData2 = class2.sample(frac=.8)
        #trainingData3 = class3.sample(frac=.8)
        #trainingData4 = class4.sample(frac=.8)
        trainingData = pd.concat([trainingData1, trainingData2], axis=0)
        #trainingData = pd.concat([trainingData1, trainingData2, trainingData3, trainingData4], axis=0)
        testData1 = class1.drop(trainingData1.index)
        testData2 = class2.drop(trainingData2.index)
        #testData3 = class3.drop(trainingData3.index)
        #testData4 = class4.drop(trainingData4.index)
        #testData = pd.concat([testData1, testData2, testData3, testData4], axis = 0)        
        testData = pd.concat([testData1, testData2], axis = 0)    

        trainingDataSample1 = trainingData.sample(frac=.5, random_state = 1)
        trainingDataSample2 = trainingData.drop(trainingDataSample1.index)

        combinedData = [trainingDataSample1, trainingDataSample2, testData]

        #Collect the best parameter over the 5 experiements, this will be 10 sets        
        testCollector = pd.concat([testCollector, da.knnValidate(combinedData, highestParameter)], axis=0)
    #print(parameterCollector)
    testAverage = testCollector.mean()

    print(f'The Average is: {testAverage}')    

  

def crossValidationKby2Regression(df, k=5):
    """
    Seperate data into 80% training and 20% test. Then, seperate the training data in half. All of these "buckets" are have an equal portion of class distribution.
    
    Parameters:
    df(dataframe): dataframe to cross validate
    k(int): defaulted to 5. Represents the number of times
    
    """
    #next break the dataframe in 2 one with each class value, this will assist in stratification 
    #THIS WILL NEED TO BE UPDATED IF MORE THAN 2 CLASSES
    testResult1 = []
    for i in range(k):
       
        trainingData1 = df.sample(frac=.8, random_state = 1)
        testData1 = df.drop(trainingData1.index)     

        trainingDataSample1 = trainingData1.sample(frac=.5, random_state = 1)
        trainingDataSample2 = trainingData1.drop(trainingDataSample1.index)

        combinedData = [trainingDataSample1, trainingDataSample2, testData1]
        #Placeholder for use in later tests
        testResult1.append(hyperParameterTuning(combinedData, regression=1))
    
    print(testResult1)