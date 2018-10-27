# coding: utf-8
'''
@author: rohangupta
'''

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import random

CONCATENATION = "Concatenation"
SUBTRACTION = "Subtraction"
HUMAN_OBSERVED = "Human Observed Dataset"
GSC = "GSC Dataset"

################################
datasetType = HUMAN_OBSERVED        #HUMAN_OBSERVED / GSC
featureSetting = SUBTRACTION        #CONCATENATION / SUBTRACTION 
doLinearReg = True                  #True / False
doLogisticReg = False                #True / False
doNeuralNet = True                  #True / False
################################

if (datasetType == HUMAN_OBSERVED):
    featuresFile = "Human/HumanObserved_Features_Data.csv"
    samePairsFile = "Human/same_pairs.csv"
    diffPairsFIle = "Human/diffn_pairs.csv"
    HALF_LENGTH_DB = 791

    if (featureSetting == CONCATENATION):
        LENGTH_FEATURES = 18
    elif (featureSetting == SUBTRACTION):
        LENGTH_FEATURES = 9

elif (datasetType == GSC):
    featuresFile = "Human/GSC_Features.csv"
    samePairsFile = "Human/same_pairs.csv"
    diffPairsFIle = "Human/diffn_pairs.csv"
    HALF_LENGTH_DB = 71531

    if (featureSetting == CONCATENATION):
        LENGTH_FEATURES = 1024
    elif (featureSetting == SUBTRACTION):
        LENGTH_FEATURES = 512

trainPercent = 80
validPercent = 10
testPercent = 10

m = 6
phi = []
epoch = 40
la = 2
eta = 0.01

trainErms = []
validErms = []
testErms = []

trainAcc = []
validAcc = []
testAcc = []


### Function for importing raw data
def importRawData(filePath):
    df = pd.read_csv(filePath)
    dataDict = {}
    for i in range(len(df)):
        key = df.iat[i, 0]
        value = []
        for j in range(1, df.shape[1]):
            value.append(int(df.iat[i, j]))

        dataDict[key] = value
    return dataDict

### Function for importing raw target
def importRawTarget(filePath):
    df = pd.read_csv(filePath)
    if (len(df) > HALF_LENGTH_DB):
        randIndices = random.sample(np.arange(len(df)), HALF_LENGTH_DB)

        randMatrix = []
        for i in range(HALF_LENGTH_DB):
            randMatrix.append(df.iloc[randIndices[i], :])
        return np.asmatrix(randMatrix)
    return np.asmatrix(df)

### Function for creating the dataset
def createDataset(dataDict, pairs, featureSetting):
    dataset = []
    for i in range(HALF_LENGTH_DB):
        row = []
        list1 = []
        list2 = []

        #row.append(pairs[i][0])
        #row.append(pairs[i][1])
        list1 = dataDict[pairs[i, 0]]
        list2 = dataDict[pairs[i, 1]]
        
        if (featureSetting == CONCATENATION):
            row = row+list1+list2
        elif (featureSetting == SUBTRACTION):
            for i in range(len(list1)):
                row.append(list1[i] - list2[i])

        row.append(int(pairs[i, 2]))
        dataset.append(row)
    return np.asmatrix(dataset)

### Function for joining and shuffling the datasets
def joinShuffleDataset(dataset1, dataset2):
    dataset = np.concatenate((dataset1, dataset2), axis=0)
    np.random.shuffle(dataset)
    return dataset

### Function for partitioning dataset
def partitionData(rawData, tr_percent, va_percent, te_percent):
    train_len = int(math.ceil(rawData.shape[0] * tr_percent * 0.01))
    valid_len = int(rawData.shape[0] * va_percent * 0.01)
    test_len = int(rawData.shape[0] * te_percent * 0.01)

    trainX = rawData[0:train_len, :]
    validX = rawData[train_len:(train_len+valid_len), :]
    testX = rawData[(train_len+valid_len):, 0]
    return trainX, validX, testX

### Function for partitioning target
def partitionTarget(rawTarget, tr_percent, va_percent, te_percent):
    train_len = int(math.ceil(np.size(rawTarget) * tr_percent * 0.01))
    valid_len = int(np.size(rawTarget) * va_percent * 0.01)
    test_len = int(np.size(rawTarget) * te_percent * 0.01)

    trainT = rawTarget[0:train_len]
    validT = rawTarget[train_len: train_len+valid_len]
    testT = rawTarget[train_len+valid_len:]
    return trainT, validT, testT

### Function for calculating Big Sigma
def calculateBigSigma(rawData, mu, tr_percent):
    #rawData = (18xn)
    bigSigma = np.zeros((rawData.shape[1], rawData.shape[1]))
    #bigSigma = (18x18)
    training_len = int(math.ceil(rawData.shape[0]*(tr_percent*0.01)))
    varVect = []

    for j in range(rawData.shape[1]):
        vct = []
        for i in range(0, training_len):
            vct.append(rawData[i, j])
        varVect.append(np.var(vct))
        #varVect = (18)

    for i in range(rawData.shape[1]):
        bigSigma[i, i] = varVect[i]
    
    bigSigma = np.dot(200, bigSigma)
    return bigSigma

### Function for calculating Radial Basis Function
def calculateBasisFn(dataRow, muRow, bigSigmaInv):
    x = np.subtract(dataRow, muRow)
    y = np.dot(bigSigmaInv, np.transpose(x))
    z = np.dot(x, y)

    phi_x = math.exp(-0.5*z)
    return phi_x

### Function for calculating Design Matrix (Phi)
def calculatePhi(rawData, mu, bigSigma):
    phi = np.zeros((len(rawData), len(mu)))
    bigSigmaInv = np.linalg.inv(bigSigma)
    for i in range(0, len(rawData)):
        for j in range(0, len(mu)):
            phi[i, j] = calculateBasisFn(rawData[i], mu[j], bigSigmaInv)

    return np.matrix(phi)

### Function for calculating Output (y)
def calculateOutput(w, phi):
    #w = (10x1), phi = (nx10)
    y = np.dot(np.transpose(w), np.transpose(phi))
    y = np.transpose(y)
    #y = (10x1)
    return y

### Function for calculating Error
def calculateErmsnAcc(y, t):
    sumE = 0.0
    count = 0
    for i in range(0,len(y)):
        sumE += (t[i] - y[i]) ** 2
        if (int(np.around(y[i], 0)) == t[i]):
            count += 1

    Erms = math.sqrt(sumE/len(t))
    Acc = float(count*100)/len(t)
    return Erms, Acc

### Sigmoid function
def sigmoid(X):
    return 1/(1 + np.exp(-X))



### Import and Prepare Dataset
humanDataDict = importRawData(featuresFile)
samePairs = importRawTarget(samePairsFile)
diffPairs = importRawTarget(diffPairsFIle)

sameDataset = createDataset(humanDataDict, samePairs, featureSetting)
diffDataset = createDataset(humanDataDict, diffPairs, featureSetting)

### Joining and Shuffling Dataset
mainDataset = joinShuffleDataset(sameDataset, diffDataset)

### Splitting Dataset into Features and Target
rawData = mainDataset[:, :LENGTH_FEATURES]
rawTarget = mainDataset[:, LENGTH_FEATURES]

print("Data Shape: ", rawData.shape)
print("Target Shape: ", rawTarget.shape)

trainData, validData, testData = partitionData(rawData, trainPercent, validPercent, testPercent)
trainTarget, validTarget, testTarget = partitionTarget(rawTarget, trainPercent, validPercent, testPercent)
print("Training data shape: ", trainData.shape, " Training target shape: ", trainTarget.shape)
print("Validation data shape: ", validData.shape, "Validation target shape: ", validTarget.shape)
print("Testing data shape: ", testData.shape, "Testing target shape: ", testTarget.shape)


### Linear Regerssion Solution
if (doLinearReg):

    #Performing k-means clustering
    kMeans = KMeans(n_clusters=m, random_state=0).fit(trainData)
    mu = kMeans.cluster_centers_

    bigSigma = calculateBigSigma(rawData, mu, trainPercent)
    print("Big Sigma shape: ", bigSigma.shape)

    trainPhi = calculatePhi(trainData, mu, bigSigma)
    print("Training Phi shape: ", trainPhi.shape)
    validPhi = calculatePhi(validData, mu, bigSigma)
    testPhi = calculatePhi(testData, mu, bigSigma) 

    #Initializing Weights
    #w = np.zeros(len(trainPhi[0]))
    wVec = [0 for i in range(m)]
    w = np.matrix(wVec).reshape((m, 1))
    #w = (10x1)
    print("Weights shape: ", w.shape)


    ### Performing Stochastic Gradient Descent
    for i in range(epoch):
        #trainTarget[i] = (1), w = (10x1), trainPhi[i] = (1x10)
        delta_Ed = - np.dot((trainTarget[i] - np.dot(np.transpose(w), np.transpose(trainPhi[i]))), trainPhi[i])
        delta_Ed = np.transpose(delta_Ed)
        la_delta_Ew = np.dot(la, w)
        delta_E = np.add(delta_Ed, la_delta_Ew)
        delta_w = - np.dot(eta, delta_E)
        w_new = np.add(w, delta_w)
        w = w_new

        trainY = calculateOutput(w, trainPhi)
        tempErms, tempAcc = calculateErmsnAcc(trainY, trainTarget)
        trainErms.append(tempErms)
        trainAcc.append(tempAcc)

        validY = calculateOutput(w, validPhi)
        tempErms, tempAcc = calculateErmsnAcc(validY, validTarget)
        validErms.append(tempErms)
        validAcc.append(tempAcc)

        print("Iteration: " + str(i+1) + ", Training Erms = " + str(trainErms[i]) + ", Validation Erms = " + str(validErms[i]))

    testY = calculateOutput(w, testPhi)
    tempErms, tempAcc = calculateErmsnAcc(testY, testTarget)
    testErms.append(tempErms)
    testAcc.append(tempAcc)

    print ('-----Linear Regression Performance using Stochastic Gradient Descent-----')
    print ("Dataset Type = " + datasetType)
    print ("Feature Setting = " + featureSetting)
    print ("m = " + str(m) + ", lambda = " + str(la) + ", eta = " + str(eta) + ", epoch = " + str(epoch))
    print ("Erms Training   = " + str(np.around(min(trainErms),5)))
    print ("Erms Validation = " + str(np.around(min(validErms),5)))
    print ("Erms Testing    = " + str(np.around(min(testErms),5)))


### Logistic Regression Solution
if(doLogisticReg):

    X_count = trainData.shape[1]

    theta = np.zeros(X_count)
































