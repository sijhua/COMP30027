import pandas as pd
import numpy as np
import math

data = pd.read_csv("train.csv", header = None)
data = data.replace(9999, np.nan)

prior_dict = {}
for label in data.iloc[:,0].unique():  
    prior_dict[label] = data.iloc[:,0].value_counts()[label]/len(data)

mean_array = []
std_array = []
for label in prior_dict.keys():
    categorised_data = data[data.iloc[:,0] == label].iloc[:,1:]
    mean_array.append(categorised_data.mean(axis = 0, skipna= True).values)
    std_array.append(categorised_data.std(axis = 0, skipna = True).values)
mean_array = np.array(mean_array)
std_array = np.array(std_array)

def calculateProbability(x, mean, std):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(std, 2))))
    return (1/(math.sqrt(2*math.pi)*std))*exponent

def calculateClassProbabilities(testVector):
    column_number = len(mean_array[0])
    probabilities = {}
    key_num = 0
    for label in prior_dict.keys():
        probabilities[label] = 1
        for i in range(column_number):
            x = testVector[i]
            probabilities[label] *= calculateProbability(x, mean_array[key_num][i], std_array[key_num][i])
        key_num+=1;
    return probabilities

list = test.values.tolist()
predictions = []
for i in range(0,len(test)):
    probabilities = calculateClassProbabilities(list[i][1:])
    bestLabel, bestProb = None, -1
    for label, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = label
    predictions.append(bestLabel)

correct = 0
for i in range(len(test)):
    if list[i][0] == predictions[i]:
        correct += 1
print(correct/float(len(test)))
