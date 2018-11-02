# coding: utf-8
'''
@author: rohangupta
'''

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import random

CONCATENATION = "Concatenation"
SUBTRACTION = "Subtraction"
HUMAN_OBSERVED = "Human Observed Dataset"
GSC = "GSC Dataset"

trainPercent = 80
validPercent = 10
testPercent = 10

### FUnction for getting values according to Dataset and Featuresetting
def getValues(datasetType, featureSetting):
    if (datasetType == HUMAN_OBSERVED):
        featuresFile = "Human/HumanObserved_Features_Data.csv"
        samePairsFile = "Human/same_pairs.csv"
        diffPairsFile = "Human/diffn_pairs.csv"
        HALF_LENGTH_DB = 791
        if (featureSetting == CONCATENATION):
            LENGTH_FEATURES = 18
            m = 7
        elif (featureSetting == SUBTRACTION):
            LENGTH_FEATURES = 9
            m = 3
    elif (datasetType == GSC):
        featuresFile = "GSC/GSC_Features.csv"
        samePairsFile = "GSC/same_pairs.csv"
        diffPairsFile = "GSC/diffn_pairs.csv"
        HALF_LENGTH_DB = 71531
        if (featureSetting == CONCATENATION):
            LENGTH_FEATURES = 1024
            m = 100
        elif (featureSetting == SUBTRACTION):
            LENGTH_FEATURES = 512
            m = 50
    return featuresFile, samePairsFile, diffPairsFile, HALF_LENGTH_DB, LENGTH_FEATURES, m

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
    print("Data Imported!")
    return dataDict

### Function for importing raw target
def importRawTarget(filePath, HALF_LENGTH_DB):
    df = pd.read_csv(filePath)
    if (len(df) > HALF_LENGTH_DB):
        randIndices = random.sample(np.arange(len(df)), HALF_LENGTH_DB)

        randMatrix = []
        for i in range(HALF_LENGTH_DB):
            randMatrix.append(df.iloc[randIndices[i], :])
        return np.asmatrix(randMatrix)
    print("Target Imported!")
    return np.asmatrix(df)

### Function for creating the dataset
def createDataset(dataDict, pairs, featureSetting, HALF_LENGTH_DB):
    dataset = []
    for i in range(0, HALF_LENGTH_DB):
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
            for j in range(len(list1)):
                row.append(abs(list1[j] - list2[j]))

        row.append(int(pairs[i, 2]))
        dataset.append(row)
    print("Dataset Created!")
    return np.asmatrix(dataset)

### Function for joining and shuffling the datasets
def joinShuffleDataset(dataset1, dataset2):
    dataset = np.concatenate((dataset1, dataset2), axis=0)
    np.random.shuffle(dataset)
    print("Dataset Joined and Shuffled!")
    return dataset

### Function for partitioning dataset
def partitionData(rawData, tr_percent, va_percent, te_percent):
    train_len = int(math.ceil(rawData.shape[0] * tr_percent * 0.01))
    valid_len = int(rawData.shape[0] * va_percent * 0.01)
    test_len = int(rawData.shape[0] * te_percent * 0.01)

    trainX = rawData[0:train_len, :]
    validX = rawData[train_len:(train_len+valid_len), :]
    testX = rawData[(train_len+valid_len):, :]
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

def importPreparePartitionData(featuresFile, samePairsFile, diffPairsFile, featureSetting, HALF_LENGTH_DB, LENGTH_FEATURES):
    ### Import and Prepare Dataset
    humanDataDict = importRawData(featuresFile)
    samePairs = importRawTarget(samePairsFile, HALF_LENGTH_DB)
    diffPairs = importRawTarget(diffPairsFile, HALF_LENGTH_DB)

    sameDataset = createDataset(humanDataDict, samePairs, featureSetting, HALF_LENGTH_DB)
    diffDataset = createDataset(humanDataDict, diffPairs, featureSetting, HALF_LENGTH_DB)

    ### Joining and Shuffling Dataset
    mainDataset = joinShuffleDataset(sameDataset, diffDataset)
    print("Dataset Shape: ", mainDataset.shape)

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
    return rawData, rawTarget, trainData, validData, testData, trainTarget, validTarget, testTarget

### Function for calculating Means (Mu)
def calculateMu(data, m):
    kMeans = KMeans(n_clusters=m, random_state=0).fit(data)
    return np.asmatrix(kMeans.cluster_centers_)

### Function for calculating Covariance Matrix (Big Sigma)
def calculateBigSigma(rawData, mu, tr_percent):
    #bigSigma = np.zeros((rawData.shape[1], rawData.shape[1]))
    training_len = int(math.ceil(rawData.shape[0]*(tr_percent*0.01)))
    varVect = []
    delIndices = []

    for j in range(rawData.shape[1]):
        vct = []
        for i in range(0, training_len):
            vct.append(rawData[i, j])

        colVar = np.var(vct)
        if (colVar == 0.0):
            delIndices.append(j)
        else:
            varVect.append(colVar)

    bigSigma = np.zeros((len(varVect), len(varVect)))
    for i in range(len(varVect)):
        bigSigma[i, i] = varVect[i]
    
    bigSigma = np.dot(200, bigSigma)
    print("Big Sigma Calculated!")
    return bigSigma, delIndices

### Function for deleting Columns with zero variance
def delZeroVarColumns(trainData, validData, testData, mu, delIndices):
    if (len(delIndices) > 0): 
        trainData = np.delete(trainData, delIndices, axis=1)
        validData = np.delete(validData, delIndices, axis=1)
        testData = np.delete(testData, delIndices, axis=1)
        mu = np.delete(mu, delIndices, axis=1)
        print("Removed " + str(len(delIndices)) + " Columns")
    return trainData, validData, testData, mu

### Function for calculating Radial Basis Function
def calculateBasisFn(dataRow, muRow, bigSigmaInv):
    x = np.subtract(dataRow, muRow)
    y = np.dot(bigSigmaInv, np.transpose(x))
    z = np.dot(x, y)
    phi_x = math.exp(-0.5*z)
    return phi_x

### Function for calculating Design Matrix (Phi)
def calculatePhi(data, mu, bigSigma):
    phi = np.zeros((len(data), len(mu)))
    bigSigmaInv = np.linalg.inv(bigSigma)
    for i in range(0, len(data)):
        for j in range(0, len(mu)):
            phi[i, j] = calculateBasisFn(data[i], mu[j], bigSigmaInv)
    print("Phi Calculated!")
    return np.matrix(phi)

### Function for calculating Output (y)
def calculateLinearOutput(w, phi):
    y = np.dot(np.transpose(w), np.transpose(phi))
    y = np.transpose(y)
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


### Function for performing Linear Regression
def performLinearRegression(datasetType, featureSetting):
    ### Setting Hyperparameters
    epochs = 400
    la = 2
    eta = 0.01

    trainErms = []
    validErms = []
    testErms = []
    trainAcc = []
    validAcc = []
    testAcc = []

    featuresFile, samePairsFile, diffPairsFile, HALF_LENGTH_DB, LENGTH_FEATURES, m = getValues(datasetType, featureSetting)
    rawData, rawTarget, trainData, validData, testData, trainTarget, validTarget, testTarget = importPreparePartitionData(featuresFile, 
                                                        samePairsFile, diffPairsFile, featureSetting , HALF_LENGTH_DB, LENGTH_FEATURES)

    mu = calculateMu(trainData, m)
    print("Mu Shape: ", mu.shape)

    bigSigma, delIndices = calculateBigSigma(rawData, mu, trainPercent)
    print("Big Sigma shape: ", bigSigma.shape)

    trainData, validData, testData, mu = delZeroVarColumns(trainData, validData, testData, mu, delIndices)

    trainPhi = calculatePhi(trainData, mu, bigSigma)
    print("Training Phi shape: ", trainPhi.shape)
    validPhi = calculatePhi(validData, mu, bigSigma)
    print("Validation Phi shape: ", validPhi.shape)
    testPhi = calculatePhi(testData, mu, bigSigma) 
    print("Testing Phi Shape: ", testPhi.shape)

    #Initializing Weights
    w = np.matrix(np.zeros(m)).reshape((m, 1))
    print("Weights shape: ", w.shape)

    ### Performing Stochastic Gradient Descent
    for i in tqdm(range(500)):
        delta_Ed = - np.dot((trainTarget[i] - np.dot(np.transpose(w), np.transpose(trainPhi[i]))), trainPhi[i])
        delta_Ed = np.transpose(delta_Ed)
        la_delta_Ew = np.dot(la, w)
        delta_E = np.add(delta_Ed, la_delta_Ew)
        delta_w = - np.dot(eta, delta_E)
        w_new = np.add(w, delta_w)
        w = w_new

        trainY = calculateLinearOutput(w, trainPhi)
        tempErms, tempAcc = calculateErmsnAcc(trainY, trainTarget)
        trainErms.append(tempErms)
        trainAcc.append(tempAcc)

        validY = calculateLinearOutput(w, validPhi)
        tempErms, tempAcc = calculateErmsnAcc(validY, validTarget)
        validErms.append(tempErms)
        validAcc.append(tempAcc)

        if (i%50 == 0):
            print("Iteration: " + str(i+1) + ", Training Erms = " + str(trainErms[i]) + ", Validation Erms = " + str(validErms[i]))

    testY = calculateLinearOutput(w, testPhi)
    tempErms, tempAcc = calculateErmsnAcc(testY, testTarget)
    testErms.append(tempErms)
    testAcc.append(tempAcc)

    print ('-----Linear Regression Performance using Stochastic Gradient Descent-----')
    print ("Dataset Type = " + datasetType)
    print ("Feature Setting = " + featureSetting)
    print ("m = " + str(m) + ", lambda = " + str(la) + ", eta = " + str(eta) + ", epochs = " + str(epochs))
    print ("Erms Training   = " + str(np.around(min(trainErms),5)))
    print ("Erms Validation = " + str(np.around(min(validErms),5)))
    print ("Erms Testing    = " + str(np.around(min(testErms),5)))

### Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

### Function for the Adding Bias Term
def addBias(trainData, validData, testData):
    tempTrainData = trainData
    trainData = np.ones((trainData.shape[0], trainData.shape[1]+1))
    trainData[:, :-1] = tempTrainData

    tempValidData = validData
    validData = np.ones((validData.shape[0], validData.shape[1]+1))
    validData[:, :-1] = tempValidData

    tempTestData = testData
    testData = np.ones((testData.shape[0], testData.shape[1]+1))
    testData[:, :-1] = tempTestData
    return trainData, validData, testData

### Function for calculating Output (y)
def calculateLogisticOutput(theta, X):
    y = np.dot(theta.transpose(), X.transpose())
    y = y.transpose()
    return sigmoid(y)


### Function for performing Logistic Regression
def performLogisticRegression(datasetType, featureSetting):
    ### Setting Hyperparameters
    epochs = 400
    eta = 0.01

    trainErms = []
    validErms = []
    testErms = []
    trainAcc = []
    validAcc = []
    testAcc = []
    
    featuresFile, samePairsFile, diffPairsFile, HALF_LENGTH_DB, LENGTH_FEATURES, m = getValues(datasetType, featureSetting)
    rawData, rawTarget, trainData, validData, testData, trainTarget, validTarget, testTarget = importPreparePartitionData(featuresFile, 
                                                        samePairsFile, diffPairsFile, featureSetting , HALF_LENGTH_DB, LENGTH_FEATURES)

    trainData, validData, testData = addBias(trainData, validData, testData)

    ### Initializing Weights
    theta = np.matrix(np.zeros(trainData.shape[1])).reshape((trainData.shape[1], 1))
    print("Weights shape: ", theta.shape)

    ### Performing Stochastic Gradient Descent
    for i in tqdm(range(epochs)):
        z = np.dot(np.transpose(theta), np.transpose(trainData))
        z = np.transpose(z)
        a = sigmoid(z)
        gradient = np.dot(np.transpose(a - trainTarget), trainData) / trainTarget.size
        gradient = gradient.transpose()
        theta -= eta * gradient

        trainY = calculateLogisticOutput(theta, trainData)
        tempErms, tempAcc = calculateErmsnAcc(trainY, trainTarget)
        trainErms.append(tempErms)
        trainAcc.append(tempAcc)

        validY = calculateLogisticOutput(theta, validData)
        tempErms, tempAcc = calculateErmsnAcc(validY, validTarget)
        validErms.append(tempErms)
        validAcc.append(tempAcc)

        print("Iteration: " + str(i+1) + ", Training Erms = " + str(trainErms[i]) + ", Validation Erms = " + str(validErms[i]))

    testY = calculateLogisticOutput(theta, testData)
    tempErms, tempAcc = calculateErmsnAcc(testY, testTarget)
    testErms.append(tempErms)
    testAcc.append(tempAcc)

    print ('-----Logistic Regression Performance using Stochastic Gradient Descent-----')
    print ("Dataset Type            = " + datasetType)
    print ("Feature Setting         = " + featureSetting)
    print ("Hyperparameters: eta = " + str(eta) + ", epochs = " + str(epochs))
    print ("Erms Training           = " + str(np.around(min(trainErms),5)))
    print ("Accuracy Training       = " + str(np.around(max(trainAcc),2)) + "%")
    print ("Erms Validation         = " + str(np.around(min(validErms),5)))
    print ("Accuracy validation     = " + str(np.around(max(validAcc),2)) + "%")
    print ("Erms Testing            = " + str(np.around(min(testErms),5)))
    print ("Accuracy Testing        = " + str(np.around(max(testAcc),2)) + "%")



### Initializing the weights to Normal Distribution
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

### Function for performing Neural Network
def performNeuralNetwork(datasetType, featureSetting):
    EPOCHS = 1000
    LR = 0.005
    batch_size = 256

    NUM_HIDDEN_NEURONS_LAYER_1 = 256
    NUM_HIDDEN_NEURONS_LAYER_2 = 128

    trainAcc = []
    validAcc = []
    testAcc = []
    
    featuresFile, samePairsFile, diffPairsFile, HALF_LENGTH_DB, LENGTH_FEATURES, m = getValues(datasetType, featureSetting)
    rawData, rawTarget, trainData, validData, testData, trainTarget, validTarget, testTarget = importPreparePartitionData(featuresFile, 
                                                        samePairsFile, diffPairsFile, featureSetting , HALF_LENGTH_DB, LENGTH_FEATURES)

    ### Defining Placeholder
    inputTensor  = tf.placeholder(tf.float32, [None, LENGTH_FEATURES])
    outputTensor = tf.placeholder(tf.float32, [None, 1])

    input_hidden_weights  = init_weights([LENGTH_FEATURES, NUM_HIDDEN_NEURONS_LAYER_1])
    #hidden_hidden_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, NUM_HIDDEN_NEURONS_LAYER_2])
    hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 1])

    hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))
    #hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer, hidden_hidden_weights))
    output_layer = tf.matmul(hidden_layer, hidden_output_weights)

    #error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))
    error_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

    training = tf.train.GradientDescentOptimizer(LR).minimize(error_function)

    prediction = tf.nn.sigmoid(output_layer)

    with tf.Session() as sess:
    
        tf.global_variables_initializer().run()
        
        for epoch in tqdm(range(int(EPOCHS))):
            
            ### Shuffling the Training Data at each epoch
            p = np.random.permutation(range(len(trainData)))
            trainData  = trainData[p]
            trainTarget = trainTarget[p]
            
            ### Starting Batch Training
            for start in range(0, len(trainData), batch_size):
                end = start + batch_size
                sess.run(training, feed_dict={inputTensor: trainData[start:end], 
                                              outputTensor: trainTarget[start:end]})

            # Training accuracy for an epoch
            trainAcc.append(np.mean(trainTarget ==
                                 np.around(sess.run(prediction, feed_dict={inputTensor: trainData,
                                                                 outputTensor: trainTarget}), 0)))

            validAcc.append(np.mean(validTarget ==
                                 np.around(sess.run(prediction, feed_dict={inputTensor: validData,
                                                                 outputTensor: validTarget}), 0)))


            if(epoch%100 == 0):
                print(trainAcc[epoch], validAcc[epoch], acc)



        # Testing
        predictedTestTarget = sess.run(prediction, feed_dict={inputTensor: testData})

    erms, acc = calculateErmsnAcc(predictedTestTarget, testTarget)
    print("Testing Accuracy: " + str(np.around(acc, 2)) + "%")



############################################################

### Linear Regerssion Solution
performLinearRegression(HUMAN_OBSERVED, CONCATENATION) 
performLinearRegression(HUMAN_OBSERVED, SUBTRACTION) 
performLinearRegression(GSC, CONCATENATION) 
performLinearRegression(GSC, SUBTRACTION)    


### Logistic Regression Solution
performLogisticRegression(HUMAN_OBSERVED, CONCATENATION)
performLogisticRegression(HUMAN_OBSERVED, SUBTRACTION)
performLogisticRegression(GSC, CONCATENATION)
performLogisticRegression(GSC, SUBTRACTION)


### Neural Network Solution
performNeuralNetwork(HUMAN_OBSERVED, CONCATENATION)
performNeuralNetwork(HUMAN_OBSERVED, SUBTRACTION)
performNeuralNetwork(GSC, CONCATENATION)
performNeuralNetwork(GSC, SUBTRACTION)

############################################################


























