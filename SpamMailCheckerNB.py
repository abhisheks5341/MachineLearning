from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import os
import collections
def make_dic (folder):
    path = "/Users/abhishek.s/Desktop/work/LearnML/DataSet-Medium/" + "chapter1/" + folder
    allWords = []
    for emailFile in os.listdir(path):
        with open(path + "/" + emailFile, "r") as fData:
            fData.readline
            fData.readline
            allWords += fData.readline().split()
    for i in range(len(allWords)):
        if len(allWords[i]) < 2:
            allWords[i] = "-1"
        elif not allWords[i].isalpha():
            allWords[i] = "-1"
    freqDic = collections.Counter(allWords)
    freqDic.pop("-1")
    return freqDic.most_common(3000)

def featureMatrixGenerator(featureList, folder):
    path = "/Users/abhishek.s/Desktop/work/LearnML/DataSet-Medium/" + "chapter1/" + folder
    examplesNum = len(os.listdir(path))
    featureMatrix = np.zeros((examplesNum, 3000))
    Lables = np.zeros((examplesNum))
    rowNum = 0
    for emailFile in os.listdir(path):
        with open(path + "/" + emailFile, "r") as fData:
            temp = fData.readline()
            temp = fData.readline()
            for emailWord in fData.readline().split():
                if emailWord in featureList:
                    featureMatrix[rowNum, featureList.index(emailWord)] += 1
        Lables[rowNum] = 0 if emailFile.startswith("sp") else 1
        rowNum+=1
    return featureMatrix, Lables

         
featureTuple = make_dic("train-mails")
featureList = []
for i in featureTuple:
    featureList.append(i[0])
#print featureList
featureMatrixTrain, LablesTrain = featureMatrixGenerator(featureList,"train-mails")
featureMatrixTest, LablesTest = featureMatrixGenerator(featureList, "test-mails")
print sum(featureMatrixTrain[0])
clf = GaussianNB()
clf.fit(featureMatrixTrain, LablesTrain)
print accuracy_score(LablesTest, clf.predict(featureMatrixTest))