import pandas as pd
import dataFunctions as funcs
import dataAlgorithms as da
import matplotlib.pyplot as plt

    #TEST DATA - custom to test specific things
if 1==0:
    csvFilePath = 'dataSource/test.data'
    testData = pd.read_csv(csvFilePath, header=None)

    #BREAST CANCER DATA - CLASSIFICATION
if 1==1:
    csvFilePath = 'dataSource/breast-cancer-wisconsin.data'
    testData = pd.read_csv(csvFilePath, names = ['sampleCodeNumber', 'clumpThickness', 'uniformityOfCellSize', 'uniformityOfCellShape', 'marginalAdhesion', 'singleEpithelialCellSize', 'bareNuclei', 'blandChromatin', 'normalNucleoi', 'mitosis', 'class'])
    testData = funcs.fillMissingWithMean(testData)
    #Check editKnn for all possible values and find where it starts to degrade
    #for i in range(len(testData)):
    #testData = da.editKnn(testData, len(testData))
        #run the validation to get the result on the edited testData
    funcs.crossValidationKby2Classification(testData) 
    

    #CAR EVALUATION DATA - CLASSIFICATION
    #buying[v-high, high, med, low] -> [4,3,2,1]
    #maint[v-high, high, med, low] -> [4,3,2,1]
    #doors[more]-> [6]
    #persons[more] -> [5]
    #lug_boot[small, med, big]-> [1,2,3]
    #safety[low, med, high] -> [1,2,3]
    #class[unacc, acc, good, v-good]->[1,2,3,4]
if 1==0:
    csvFilePath = 'dataSource/car(1)-1.data'
    testData = pd.read_csv(csvFilePath, names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
    print(testData)
    testData = funcs.replaceOrdinalslWithIntegers(testData, ['vhigh', 'high', 'med', 'low'], [4,3,2,1], 'buying')
    testData = funcs.replaceOrdinalslWithIntegers(testData, ['vhigh', 'high', 'med', 'low'], [4,3,2,1], 'maint')
    testData = funcs.replaceOrdinalslWithIntegers(testData, ['5more'], [6], 'doors')
    testData = funcs.replaceOrdinalslWithIntegers(testData, ['more'], [5], 'persons')
    testData = funcs.replaceOrdinalslWithIntegers(testData, ['small', 'med', 'big'], [1,2,3], 'lug_boot')
    testData = funcs.replaceOrdinalslWithIntegers(testData, ['low', 'med', 'high'], [1,2,3], 'safety')
    testData = funcs.replaceOrdinalslWithIntegers(testData, ['unacc', 'acc', 'good', 'vgood'], [1,2,3,4], 'class')
    print(testData)
    funcs.crossValidationKby2Classification(testData)

    #Congressional Vote Data - Classification
    #class[democrat, republican] -> [0,1]
    #ALL OTHER COLUMNS[y,n] -> [1,0]
if 1==0:
    csvFilePath = 'dataSource/house-votes-84(1)-1.data'
    testData = pd.read_csv(csvFilePath, names = ['class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'])

    #Abalone - Regression (PREDICT RINGS)
    #SEX[M,F,I] -> sex_m, sex_f, sex_i (one hot encode)
if 1==0:
    csvFilePath = 'dataSource/abalone(1)-1.data'
    testData = pd.read_csv(csvFilePath, names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'])
    print(testData)
    testData = funcs.oneHotEncodeColumn(testData, ['Sex'])
    testData = funcs.fillMissingWithMean(testData)
    testData = funcs.zScoreStandardize(testData, 'Rings')
    print(testData)


    #Computer Hardware - Regression (Predict ERP - Estimated Relative Performance)
    #vendor[30 names] -> need to one hot encode
    #model -> drop this column
if 1==0:
    csvFilePath = 'dataSource/machine(1).data'
    testData = pd.read_csv(csvFilePath, names = ['vendor', 'Model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'])

    #Forest Fire Data - Regression (predict area)
    #month[jan - dec] -> [1..12]
    #day[mon-fri] -> [1..7]
if 1==0:
    csvFilePath = 'dataSource/forestfires(1)-1.data'
    testData = pd.read_csv(csvFilePath, names = ['xAxis', 'yAxis', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area'])
    print(testData)
    testData = funcs.replaceOrdinalslWithIntegers(testData, ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],[1,2,3,4,5,6,7,8,9,10,11,12], 'month')
    testData = funcs.replaceOrdinalslWithIntegers(testData, ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'],[1,2,3,4,5,6,7], 'day')
    testData = funcs.discretizeEqualFrequency(testData, 'wind', 4)
    print(testData)
    funcs.crossValidationKby2Regression(testData)





        
    