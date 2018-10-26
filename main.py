# coding: utf-8
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import random

CONCAT = "concatenate"
SUBTR = "subtract"
HUMAN = "human"
HALF_LEN_HUMAN = 791

dataset = HUMAN
setting = CONCAT
trainPercent = 80
validPercent = 10
testPercent = 10

m = 8
lambda_c = 0.03
phi = []
epoch = 100
La = 2
eta = 0.01

trainErms = []
validErms = []
testErms = []

trainAcc = []
validAcc = []
testAcc = []

### Function for importing raw data
def importRawData(filePath):
    dataMatrix = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(column)
            dataMatrix.append(dataRow)

        dataMatrix = dataMatrix[1:]
        dataDict = {}
        for i in range(len(dataMatrix)):
            key = dataMatrix[i][1]
            value = []

            for j in range(2,11):
                value.append(int(dataMatrix[i][j]))

            dataDict[key] = value

    return dataDict

### Function for importing raw target
def importRawTarget(filePath):
    targetMatrix = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:   
            targetRow = []
            for column in row:
                targetRow.append(column)
            targetMatrix.append(targetRow)

        targetMatrix = targetMatrix[1:]

        if (len(targetMatrix) > HALF_LEN_HUMAN):
            randIndices = random.sample(np.arange(len(targetMatrix)), HALF_LEN_HUMAN)

            randMatrix = []
            for i in range(HALF_LEN_HUMAN):
                randMatrix.append(targetMatrix[randIndices[i]])
            return randMatrix

        return targetMatrix

### Function for creating the dataset
def createDataset(dataDict, pairs, setting):
    dataset = []

    if (setting == CONCAT):
        for i in range(HALF_LEN_HUMAN):
            row = []
            list1 = []
            list2 = []

            #row.append(pairs[i][0])
            #row.append(pairs[i][1])

            list1 = dataDict[pairs[i][0]]
            list2 = dataDict[pairs[i][1]]
            row = row+list1+list2

            row.append(int(pairs[i][2]))
            dataset.append(row)

    return np.asmatrix(dataset)

### Function for joining and shuffling the datasets
def joinShuffleDataset(dataset1, dataset2):
    dataset = np.concatenate((dataset1, dataset2), axis=0)
    np.random.shuffle(dataset)

    dataset = np.transpose(dataset)
    #dataset = (8x1582)
    return dataset

### Function for partitioning dataset
def partitionData(rawData, tr_percent, va_percent, te_percent):
    train_len = int(math.ceil(np.size(rawData, 1) * tr_percent * 0.01))
    valid_len = int(np.size(rawData, 1) * va_percent * 0.01)
    test_len = int(np.size(rawData, 1) * te_percent * 0.01)

    trainX = rawData[:, 0:train_len]
    validX = rawData[:, train_len:(train_len+valid_len)]
    testX = rawData[:, (train_len+valid_len):]
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
    bigSigma = np.zeros((np.size(rawData, 0), np.size(rawData, 0)))
    #bigSigma = (18x18)
    rawDataT = np.transpose(rawData)
    #rawDataT = (nx18)
    training_len = int(math.ceil(np.size(rawDataT, 0)*(tr_percent*0.01)))
    varVect = []

    for i in range(0, np.size(rawDataT, 1)):
        vct = []
        for j in range(0, training_len):
            vct.append(rawData[i, j])
        varVect.append(np.var(vct))
        #varVect = (18)

    for i in range(len(rawData)):
        bigSigma[i, i] = varVect[i]
    
    ###bigSigma = np.dot(200, bigSigma)
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
    #rawDataT = [nx41]
    rawDataT = np.transpose(rawData)
    phi = np.zeros((len(rawDataT), len(mu)))
    bigSigmaInv = np.linalg.inv(bigSigma)
    for i in range(0, len(rawDataT)):
        for j in range(0, len(mu)):
            phi[i, j] = calculateBasisFn(rawDataT[i], mu[j], bigSigmaInv)

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

### Import and Prepare Dataset
humanDataDict = importRawData("Human/HumanObserved_Features_Data.csv")
samePairs = importRawTarget("Human/same_pairs.csv")
diffPairs = importRawTarget("Human/diffn_pairs.csv")

sameDataset = createDataset(humanDataDict, samePairs, CONCAT)
diffDataset = createDataset(humanDataDict, diffPairs, CONCAT)

mainDataset = joinShuffleDataset(sameDataset, diffDataset)


rawData = mainDataset[:18, :]
rawTarget = mainDataset[18, :]
rawTarget = rawTarget.transpose()

print("Data Shape: ", rawData.shape)
print("Target Shape: ", rawTarget.shape)

trainData, validData, testData = partitionData(rawData, trainPercent, validPercent, testPercent)
trainTarget, validTarget, testTarget = partitionTarget(rawTarget, trainPercent, validPercent, testPercent)
print("Training data shape: ", trainData.shape, " Training target shape: ", trainTarget.shape)
print("Validation data shape: ", validData.shape, "Validation target shape: ", validTarget.shape)
print("Testing data shape: ", testData.shape, "Testing target shape: ", testTarget.shape)


#Performing k-means clustering
kMeans = KMeans(n_clusters=m, random_state=0).fit(np.transpose(trainData))
mu = kMeans.cluster_centers_


bigSigma = calculateBigSigma(rawData, mu, trainPercent)
print("Big Sigma shape:", bigSigma.shape)

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


### Performing Gradient Descent
for i in range(epoch):
    #trainTarget[i] = (1), w = (10x1), trainPhi[i] = (1x10)
    delta_Ed = - np.dot((trainTarget[i] - np.dot(np.transpose(w), np.transpose(trainPhi[i]))), trainPhi[i])
    delta_Ed = np.transpose(delta_Ed)
    #delta_Ed = (10x1)
    la_delta_Ew = np.dot(La, w)
    #la_delta_Ew = (10x1)
    delta_E = np.add(delta_Ed, la_delta_Ew)
    #delta_E = (10x1)
    delta_w = - np.dot(eta, delta_E)
    #delta_w = (10x1)
    w_new = np.add(w, delta_w)
    #w_new = (10x1)
    w = w_new

    trainY = calculateOutput(w, trainPhi)
    #trainY = (1266,1)

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

print ('----------Gradient Descent Solution--------------------')
print ("m = " + str(m) + ", lambda = " + str(lambda_c) + ", eta = " + str(eta) + ", epoch = " + str(epoch))
print ("Erms Training   = " + str(np.around(min(trainErms),5)))
print ("Erms Validation = " + str(np.around(min(validErms),5)))
print ("Erms Testing    = " + str(np.around(min(testErms),5)))































