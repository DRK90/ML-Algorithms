import pandas as pd
import dataAlgorithms as da
import dataAlgorithmsRegression as dar
import decisionTreeAlgorithms as trees
import DecTreeReg as regTrees

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

def crossValidationKby2Classification(df, k=1):
    """
    Seperate data into 80% training and 20% test. Then, seperate the training data in half. All of these "buckets" are have an equal portion of class distribution.
    
    Parameters:
    df(dataframe): dataframe to cross validate
    k(int): defaulted to 5. Represents the number of times
    
    """
    #next break the dataframe in 2 one with each class value, this will assist in stratification 
    #EDIT KNN will be done first

    #First find the root node
    if 1==1:
        for i in range(1):
            class1 = df[df['class']==4]
            class2 = df[df['class']==2]
            #class3 = df[df['class']==2]
            #class4 = df[df['class']==1]
            trainingData1 = class1.sample(frac=.8)
            trainingData2 = class2.sample(frac=.8)
            #trainingData3 = class3.sample(frac=.8)
            #trainingData4 = class4.sample(frac=.8)
            trainingData = pd.concat([trainingData1, trainingData2], axis=0)
            trainingData = trainingData.sample(frac=1).reset_index(drop=True)
            #trainingData = pd.concat([trainingData1, trainingData2, trainingData3, trainingData4], axis=0)
            testData1 = class1.drop(trainingData1.index)
            testData2 = class2.drop(trainingData2.index)
            #testData3 = class3.drop(trainingData3.index)
            #testData4 = class4.drop(trainingData4.index)
            #testData = pd.concat([testData1, testData2, testData3, testData4], axis = 0)        
            testData = pd.concat([testData1, testData2], axis = 0)    
            trainingDataSample1 = trainingData.sample(frac=.5)
            trainingDataSample2 = trainingData.drop(trainingDataSample1.index)

            #recurseive version
            root = trees.calculateRoot(trainingDataSample1)

            trees.printTree(root)

            predictions = []
            actuals = testData["class"].tolist()


            for i, row in testData.iterrows():
                prediction = trees.predict(row, root)  
                predictions.append(prediction)

            correctPredictions = 0
            for i in range(len(predictions)):
                if predictions[i] == actuals[i]:
                    correctPredictions += 1

            accuracy = correctPredictions / len(actuals) * 100
            print(f"Pre-pruning Accuracy: {accuracy:.2f}%")


            #NOW WE PRUNE AND DO IT AGAIN
            if 1==1:
                trees.prune(root, trainingDataSample2, root)
                trees.printTree(root)


                predictions = []
                actuals = testData["class"].tolist()


                for i, row in testData.iterrows():
                    prediction = trees.predict(row, root)  
                    predictions.append(prediction)

                correctPredictions = 0
                for i in range(len(predictions)):
                    if predictions[i] == actuals[i]:
                        correctPredictions += 1

                accuracy = correctPredictions / len(actuals) * 100
                print(f"Post-pruning Accuracy: {accuracy:.2f}%")


            #first root (NOT RECURSIVE)
            if 1==0:
                rootValue = trees.calculateRoot(trainingData)
                if type(rootValue)==int:
                    print(f"trainingData leaf: {rootValue}")
                else:
                    feature = rootValue[0]['bestFeature']
                    value = rootValue[0]['bestValue']
                    trainingDataLeft = trainingData[trainingData[feature] <= value]
                    trainingDataRight = trainingData[trainingData[feature] > value]
                    print(f"trainingData: {rootValue}")

                    #first left child
                    rootValue = trees.calculateRoot(trainingDataLeft)
                    if type(rootValue)==int:
                        print(f"trainingDataLeft leaf: {rootValue}")
                    else:
                        feature = rootValue[0]['bestFeature']
                        value = rootValue[0]['bestValue']
                        trainingDataLeftLeft = trainingDataLeft[trainingDataLeft[feature] <= value]
                        trainingDataLeftRight = trainingDataLeft[trainingDataLeft[feature] > value]
                        print(f"trainingDataLeft: {rootValue}")

                        # left child from first left child
                        rootValue = trees.calculateRoot(trainingDataLeftLeft)
                        if type(rootValue)==int:
                            print(f"trainingDataLeftLeft leaf: {rootValue}")
                        else:            
                            feature = rootValue[0]['bestFeature']
                            value = rootValue[0]['bestValue']
                            trainingDataLeftLeftLeft = trainingDataLeftLeft[trainingDataLeftLeft[feature] <= value]
                            trainingDataLeftLeftRight = trainingDataLeftLeft[trainingDataLeftLeft[feature] > value]
                            print(f"trainingDataLeftLeft: {rootValue}")

                            # left child from left child from first left child
                            rootValue = trees.calculateRoot(trainingDataLeftLeftLeft)
                            if type(rootValue)==int:
                                print(f"trainingDataLeftLeftLeft leaf: {rootValue}")
                            else:
                                feature = rootValue[0]['bestFeature']
                                value = rootValue[0]['bestValue']
                                trainingDataLeftLeftLeftLeft = trainingDataLeftLeftLeft[trainingDataLeftLeftLeft[feature] <= value]
                                trainingDataLeftLeftLeftRight = trainingDataLeftLeftLeft[trainingDataLeftLeftLeft[feature] > value]
                                print(f"trainingDataLeftLeftLeft: {rootValue}")

                                # left child from left child from left child from first left child
                                rootValue = trees.calculateRoot(trainingDataLeftLeftLeftLeft)
                                if type(rootValue)==int:
                                    print(f"trainingDataLeftLeftLeftLeft leaf: {rootValue}")
                                else:
                                    feature = rootValue[0]['bestFeature']
                                    value = rootValue[0]['bestValue']
                                    trainingDataLeftLeftLeftLeftLeft = trainingDataLeftLeftLeftLeft[trainingDataLeftLeftLeftLeft[feature] <= value]
                                    trainingDataLeftLeftLeftLeftRight = trainingDataLeftLeftLeftLeft[trainingDataLeftLeftLeftLeft[feature] > value]
                                    print(f"trainingDataLeftLeftLeftLeft: {rootValue}")

                                # right child from left child from left child from first left child
                                rootValue = trees.calculateRoot(trainingDataLeftLeftLeftRight)
                                if type(rootValue)==int:
                                    print(f"trainingDataLeftLeftLeftRight leaf: {rootValue}")
                                else:
                                    feature = rootValue[0]['bestFeature']
                                    value = rootValue[0]['bestValue']
                                    trainingDataLeftLeftLeftRightLeft = trainingDataLeftLeftLeftRight[trainingDataLeftLeftLeftRight[feature] <= value]
                                    trainingDataLeftLeftLeftRightRight = trainingDataLeftLeftLeftRight[trainingDataLeftLeftLeftRight[feature] > value]
                                    print(f"trainingDataLeftLeftLeftRight: {rootValue}")

                            # right child from left child from first left child
                            rootValue = trees.calculateRoot(trainingDataLeftLeftRight)
                            if type(rootValue)==int:
                                print(f"trainingDataLeftLeftRight leaf: {rootValue}")
                            else:            
                                feature = rootValue[0]['bestFeature']
                                value = rootValue[0]['bestValue']
                                trainingDataLeftLeftRightLeft = trainingDataLeftLeft[trainingDataLeftLeft[feature] <= value]
                                trainingDataLeftLeftRightRight = trainingDataLeftLeft[trainingDataLeftLeft[feature] > value]
                                print(f"trainingDataLeftLeftRight: {rootValue}")

                        # right child from first left child
                        rootValue = trees.calculateRoot(trainingDataLeftRight)
                        if type(rootValue)==int:
                            print(f"trainingDataLeftRight leaf: {rootValue}")
                        else:
                            feature = rootValue[0]['bestFeature']
                            value = rootValue[0]['bestValue']
                            trainingDataLeftRightLeft = trainingDataLeftRight[trainingDataLeftRight[feature] <= value]
                            trainingDataLeftRightRight = trainingDataLeftRight[trainingDataLeftRight[feature] > value]
                            print(f"trainingDataLeftRight: {rootValue}")

                            # left child from right child from first left child
                            rootValue = trees.calculateRoot(trainingDataLeftRightLeft)
                            if type(rootValue)==int:
                                print(f"trainingDataLeftRightLeft leaf: {rootValue}")
                            else:
                                feature = rootValue[0]['bestFeature']
                                value = rootValue[0]['bestValue']
                                trainingDataLeftRightLeftLeft = trainingDataLeftRightLeft[trainingDataLeftRightLeft[feature] <= value]
                                trainingDataLeftRightLeftRight = trainingDataLeftRightLeft[trainingDataLeftRightLeft[feature] > value]
                                print(f"trainingDataLeftRightLeft: {rootValue}")

                            # right child from right child from first left child
                            rootValue = trees.calculateRoot(trainingDataLeftRightRight)
                            if type(rootValue)==int:
                                print(f"trainingDataLeftRightRight leaf: {rootValue}")
                            else:
                                feature = rootValue[0]['bestFeature']
                                value = rootValue[0]['bestValue']
                                trainingDataLeftRightRightLeft = trainingDataLeftRightRight[trainingDataLeftRightRight[feature] <= value]
                                trainingDataLeftRightRightRight = trainingDataLeftRightRight[trainingDataLeftRightRight[feature] > value]
                                print(f"trainingDataLeftRightRight: {rootValue}")

                    #first right child
                    rootValue = trees.calculateRoot(trainingDataRight)
                    if type(rootValue)==int:
                        print(f"trainingDataRight leaf: {rootValue}")
                    else:
                        feature = rootValue[0]['bestFeature']
                        value = rootValue[0]['bestValue']
                        trainingDataRightLeft = trainingDataRight[trainingDataRight[feature] <= value]
                        trainingDataRightRight = trainingDataRight[trainingDataRight[feature] > value]
                        print(f"trainingDataRight: {rootValue}")

                        #left child from first right child
                        rootValue = trees.calculateRoot(trainingDataRightLeft)
                        if type(rootValue)==int:
                            print(f"trainingDataRightLeft leaf: {rootValue}")
                        else:
                            feature = rootValue[0]['bestFeature']
                            value = rootValue[0]['bestValue']
                            trainingDataRightLeftLeft = trainingDataRightLeft[trainingDataRightLeft[feature] <= value]
                            trainingDataRightLeftRight = trainingDataRightLeft[trainingDataRightLeft[feature] > value]
                            print(f"trainingDataRightLeft: {rootValue}")

                        #right child from first right child
                        rootValue = trees.calculateRoot(trainingDataRightRight)
                        if type(rootValue)==int:
                            print(f"trainingDataRightRight leaf: {rootValue}")
                        else:
                            feature = rootValue[0]['bestFeature']
                            value = rootValue[0]['bestValue']
                            trainingDataRightRightLeft = trainingDataRightRight[trainingDataRightRight[feature] <= value]
                            trainingDataRightRightRight = trainingDataRightRight[trainingDataRightRight[feature] > value]
                            print(f"trainingDataRightRight: {rootValue}")

       
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
        testData = pd.concat([testData1, testData2], axis = 0)    

        trainingDataSample1 = trainingData.sample(frac=.5, random_state = 1)
        trainingDataSample2 = trainingData.drop(trainingDataSample1.index)






  

def crossValidationKby2Regression(df, k=1):
    """
    Seperate data into 80% training and 20% test. Then, seperate the training data in half. All of these "buckets" are have an equal portion of class distribution.
    
    Parameters:
    df(dataframe): dataframe to cross validate
    k(int): defaulted to 5. Represents the number of times
    
    """
    #next break the dataframe in 2 one with each class value, this will assist in stratification 
    #THIS WILL NEED TO BE UPDATED IF MORE THAN 2 CLASSES

    for i in range(k):
       
        trainingData1 = df.sample(frac=.8)
        testData1 = df.drop(trainingData1.index)     

        trainingDataSample1 = trainingData1.sample(frac=.5)
        trainingDataSample2 = trainingData1.drop(trainingDataSample1.index)

        root = regTrees.calculateRoot(trainingDataSample1)
        regTrees.printTree(root)

        predictions = []
        actuals = testData1["Rings"].tolist()


        for i, row in testData1.iterrows():
            prediction = regTrees.predict(row, root)  
            predictions.append(prediction)

        correctPredictions = 0
        for i in range(len(predictions)):
            if predictions[i] == actuals[i]:
                correctPredictions += 1

        accuracy = correctPredictions / len(actuals) * 100
        print(f"Pre-pruning Accuracy: {accuracy:.2f}%")       

        #prune
        regTrees.prune(root, trainingDataSample2, root)
        regTrees.printTree(root)

        predictions = []
        actuals = testData1["Rings"].tolist()


        for i, row in testData1.iterrows():
            prediction = regTrees.predict(row, root)  
            predictions.append(prediction)

        correctPredictions = 0
        for i in range(len(predictions)):
            if predictions[i] == actuals[i]:
                correctPredictions += 1

        accuracy = correctPredictions / len(actuals) * 100
        print(f"Post-pruning Accuracy: {accuracy:.2f}%")  

    
    