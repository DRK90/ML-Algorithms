import pandas as pd
import numpy as num
import warnings
#ignoring future warnings for concatanating dataframes to empty dataframes.  In the future I can address it.
warnings.filterwarnings('ignore', category=FutureWarning)

def knnTest(data):
    """
    Performs k nearest neighbors on a dataset that is already broken into train1 (40%) train2 (40%) and test (20%) stratified partitions

    Parameters:
    data(array): array of the 3 dataframes: train, test, validate

    Returns:
    overallResultsForAllTests(dataframe): The results of the two tests, one for each test set, on each of the parameters
    """
    #get the list of features we will use to measure the distance
    attributes = data[0].drop(columns=['Rings']).columns.tolist()
    #capture the results of the test to get the accuracy
    overallResultsForEachTest = pd.DataFrame(columns=['k', 'gamma', 'epsilon', 'accuracy'])

    #candidate paramater choices
    gammaChoices = [.01,.1,1,10]
    epsilonChoices = [0.25,0.5,1,2]
    kChoices = [3,5,7]

    #run the loop twice, flip it to test on the opposite datasets
    for i in range(2):
        #run the loop for each of the k choices
        for k in kChoices:
            for e in epsilonChoices:
                for gamma in gammaChoices:
                    #results is the list of correct answers in each iteration of a particular k value
                    results = []
                    #iterate over each instance in the test set (one of the 40% slices)
                    for index, instance in data[2].iterrows():
                        #calculate the euclidean distance of each point in the test set against the training set
                        data[i]['distance'] = num.sqrt(((data[i][attributes] - instance[attributes])**2).sum(axis=1))
                        #find the k points with the smallest distance
                        nearestNeighbors = data[i].nsmallest(k,'distance')
                        #of the nearest neighbor list predict based off the mode
                        weights = num.exp(-gamma*nearestNeighbors['distance']**2)
                        #predict number of rings using the weights calculated
                        valuePrediction = num.average(nearestNeighbors['Rings'], weights=weights)
                        #add a 1 or 0 to results if it was a correct prediction based on e value
                        if ((instance['Rings'] - e) < valuePrediction) and ((instance['Rings'] + e) >= valuePrediction ):
                            results.append(1)
                        else:
                            results.append(0)
                    #Percent correct is the percent this particular parameter set value got this time around
                    percentCorrect = sum(results) / len(results)
                    #print(f'k = {k} was {percentCorrect}% accurate')
                    #This list will only be 4*4*4 long, 1 for each parameter pair.
                    tempDf = pd.DataFrame({
                        'k': [k],
                        'gamma': [gamma],
                        'epsilon': [e],
                        'accuracy': percentCorrect
                    })
                    print(tempDf)
                    overallResultsForEachTest= pd.concat([overallResultsForEachTest, tempDf], ignore_index=True)
        #print(f'k=1 {overallResultsForEachTest[0]}, k=3 {overallResultsForEachTest[1]}, k=5 {overallResultsForEachTest[2]}, k=7 {overallResultsForEachTest[3]} ')
        #Add the list of the 4 k value percentages for each pass.  There will be 2 of these lists added
        #overallResultsForAllTests.loc[len(overallResultsForAllTests)] = overallResultsForEachTest
        #reset the overallResults variable to load 4 new ones on the next pass
        #overallResultsForEachTest = []

    #print(overallResultsForAllTests)
    return overallResultsForEachTest
    #print(overallResultsForAllTests)

def knnValidate(data, k):
    """
    Performs k nearest neighbors on a dataset that is already broken into train1 (40%) train2 (40%) and test (20%) stratified partitions
    This version will use the known best parameter to get results
    
    Parameters:
    data(array): array of the 3 dataframes: train, test, validate
    k(int): known parameter

    Returns: none
    """
    #get the list of features we will use to measure the distance
    attributes = data[0].drop(columns=['class', 'sampleCodeNumber']).columns.tolist()
    #capture the results of the test to get the accuracy
    overallResultsForEachTest = []
    overallResultsForAllTests = pd.DataFrame(columns=['best parameter'])

    #run the loop twice, flip it to test on the opposite datasets
    for i in range(2):
        #results is the list of correct answers in each iteration of a particular k value
        results = []
        #iterate over each instance in the test set (one of the 40% slices)
        for index, instance in data[1-i].iterrows():
            #calculate the euclidean distance of each point in the test set against the training set
            data[i]['distance'] = num.sqrt(((data[i][attributes] - instance[attributes])**2).sum(axis=1))
            #find the k points with the smallest distance
            nearestNeighbors = data[i].nsmallest(k,'distance')
            #of the nearest neighbor list predict based off the mode
            classPrediction = nearestNeighbors['class'].mode()[0]
            #add a 1 or 0 to results if it was a correct prediction
            results.append(1 if classPrediction == instance['class'] else 0)
        #Percent correct is the percent this particular K value got this time around
        percentCorrect = sum(results) / len(results)
        #print(f'k = {k} was {percentCorrect}% accurate')
        #This list will only be 1 long, 1 for each K value.
        overallResultsForEachTest.append(percentCorrect)
        #print(f'k=1 {overallResultsForEachTest[0]}, k=3 {overallResultsForEachTest[1]}, k=5 {overallResultsForEachTest[2]}, k=7 {overallResultsForEachTest[3]} ')
        #Add the list of the 1 k value percentages for each pass.  There will be 2 of these lists added
        overallResultsForAllTests.loc[len(overallResultsForAllTests)] = overallResultsForEachTest
        overallResultsForEachTest = []

    #print(overallResultsForAllTests)
    return overallResultsForAllTests
    #print(overallResultsForAllTests)

def editKnn(df, pointsToDrop):
    """
    delete increasing number of data points that are not predicted correctly by their 1 nearest neighbor. Stop when the result gets below 20% from start

    Parameters:
    df(DateFrame): the dataframe that will be edited

    Returns: copy of the dataframe
    """  
    attributes = df.drop(columns=['class', 'sampleCodeNumber']).columns.tolist()
    #intialize a drop column
    df['drop'] = 0
    for i in range(pointsToDrop):
        point = df.iloc[i]
        #temporarily drop the point so it doesnt compare to itself
        tempDf = df.drop(df.index[i], errors='ignore')
        #calculate the euclidean distance of each point in the test set against the training set
        tempDf['distance'] = num.sqrt(((tempDf[attributes] - point[attributes])**2).sum(axis=1))
        #find the 1 points with the smallest distance
        nearestNeighbors = tempDf.nsmallest(1,'distance')
        #of the nearest neighbor list predict based off the mode
        classPrediction = nearestNeighbors['class'].mode()[0]
        #if the prediction is wrong, mark it to be dropped
        df.at[df.index[i], 'drop'] = 1 if classPrediction != point['class'] else 0
    
    #find the indices where the drop column is one
    indexOfPointsToDrop = df[df['drop'] == 1].index
    #drop the points
    df = df.drop(indexOfPointsToDrop)
    df.drop('drop', axis=1, inplace=True)
    print(f'Checked {pointsToDrop} data points. Dropped {len(indexOfPointsToDrop)} points.')
    return df

def condensedKnn(df):
    """
    start with an empty condensed set, then add points until no more are added

    Parameters:
    df(DataFrame): dataframe to condense

    Returns:
    df(DataFrame): copy of dataFrame that has been condensed
    """
    attributes = df.drop(columns=['class', 'sampleCodeNumber']).columns.tolist()
    #initialize a new dataframe with columns that match the first
    newDf = pd.DataFrame(columns=df.columns)
    #copy the first row from df to newDf, then drop it from df
    firstDataPoint = df.iloc[0]
    newDf = pd.concat([newDf, pd.DataFrame([firstDataPoint])], ignore_index=True)
    df = df.drop(df.index[0])
    #flag to indicate if a point moved - meaning we run through it all again
    pointMoved = True

    while pointMoved:
        pointMoved = False
        pointsToDrop = []
        for index, instance in df.iterrows():
            newDf['distance'] = num.sqrt(((newDf[attributes] - instance[attributes])**2).sum(axis=1))
            nearestNeighbors = newDf.nsmallest(1,'distance')
            classPrediction = nearestNeighbors['class'].mode()[0]
            #if the 1NN predicts incorrectly, we move the point from df to the newDf
            if classPrediction != instance['class']:
                dataPointToMove = instance
                newDf = pd.concat([newDf, pd.DataFrame([dataPointToMove])], ignore_index=True)                
                pointsToDrop.append(index)
                pointMoved = True
                print('Point Moved')
        df = df.drop(pointsToDrop)
    print(f'newDf is {len(newDf)}')
    newDf.drop('distance', axis=1, inplace=True)

    return newDf
    



