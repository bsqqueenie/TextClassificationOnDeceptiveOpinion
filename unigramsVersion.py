import os
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from string import digits


#-------------1=True 0=False-----------------#

pathOfNegativeReviews = 'op_spam_v1.4/negative_polarity'


def removeNumbers(text):
    remove_digits = str.maketrans('', '', digits)
    return text.translate(remove_digits)


def sentenceProcessing(sentence):
    tempList = []
    stop_words = list(get_stop_words('en'))  # About 900 stopwords
    nltk_words = list(stopwords.words('english'))  # About 150 stopwords
    stop_words.extend(nltk_words)
    stop_words.append("th")
    sentence=removeNumbers(sentence)
    for i in string.punctuation:
        sentence = sentence.replace(i, '')
    lowerCase = sentence.lower().split()
    for eachWord in lowerCase:
        if eachWord not in stop_words:
            tempList.append(eachWord)
    # print(tempList)
    return tempList


def getTheData(path):
    # remove all the stopwords
    n = 0
    trueReview = 1
    falseRevie = 0
    tempDic = {}
    trainingList = []
    trainingLabel = []
    testList = []
    testLabel = []
    for fpathe, dirs, fs in os.walk(path):
        if len(fs) > 1:
            for name in fs:
                if "fold1" in fpathe or "fold2" in fpathe or "fold3" in fpathe or "fold4" in fpathe:
                    with open(os.path.join(fpathe, name), "r") as textFile:
                        if "truthful" in fpathe:
                            for eachOne in textFile.readlines():
                                trainingLabel.append(trueReview)
                                trainingList.append(sentenceProcessing(eachOne))
                        else:
                            for eachOne in textFile.readlines():
                                trainingLabel.append(falseRevie)
                                trainingList.append(sentenceProcessing(eachOne))
                    tempDic["textList"] = trainingList
                    tempDic["labelList"] = trainingLabel
                    trainingData = pd.DataFrame(tempDic)
                    # trainingData.to_csv('trainingData.csv')
                    # trainingData.to_csv('trainingData_withoutStop.csv')
                else:
                    with open(os.path.join(fpathe, name), "r") as textFile:
                        if "truthful" in fpathe:
                            for eachOne in textFile.readlines():
                                testLabel.append(trueReview)
                                testList.append(sentenceProcessing(eachOne))
                        else:
                            for eachOne in textFile.readlines():
                                testLabel.append(falseRevie)
                                testList.append(sentenceProcessing(eachOne))
                    tempDic["textList"] = testList
                    tempDic["labelList"] = testLabel
                    testData = pd.DataFrame(tempDic)
                    # testData.to_csv('testData.csv')
    return trainingData, testData


def wordOfBag(tokenizedText, numOfFeatures):
    n=1
    dic = {}
    sortedDic = {}
    for rowNum in range(tokenizedText.shape[0]):
        for eachWord in tokenizedText.iloc[rowNum, 0]:
            if not dic.__contains__(eachWord):
                dic[eachWord] = 1
                n=n+1 #统计一共有多少词
            else:
                dic[eachWord] = dic[eachWord] + 1
    #print("the total number of words:",n)
    sortedList = sorted(dic.items(), key=lambda k: k[1], reverse=True)  # 将字典根据出现的词频大小排序
    sortedList = sortedList[0:numOfFeatures]
    for i in sortedList:
        sortedDic[i[0]] = i[1]
    return sortedDic


def word2Vector(tokenizedText, allWords):
    bagofWordList = list(allWords)
    for rowNum in range(tokenizedText.shape[0]):
        vectorList = []
        for i in bagofWordList:
            if i in tokenizedText.iloc[rowNum, 0]:
                vectorList.append(1)
            else:
                vectorList.append(0)
        tokenizedText.at[rowNum, "textList"] = vectorList

    return tokenizedText


def dataFrame2array(nameOfColumn, nameofDataFrame):
    temlist = []
    for i in nameofDataFrame[nameOfColumn]:
        temlist.append(i)
    finalArray = np.array(temlist)

    return finalArray


trainingData, testData = getTheData(pathOfNegativeReviews)
# print(trainingData,testData)
# print(trainingData)
totallFeatures=6959
numOfFeatures = 500 # 选出前N个作为特征值
bagOfWords = wordOfBag(trainingData, numOfFeatures)
#print(bagOfWords)
vectorDataofTraining = word2Vector(trainingData, bagOfWords)
vectorDataofTest = word2Vector(testData, bagOfWords)
X_train = dataFrame2array("textList", vectorDataofTraining)
Y_train = np.array(vectorDataofTraining["labelList"])
X_test = dataFrame2array("textList", vectorDataofTest)
Y_test = np.array(vectorDataofTest["labelList"])
# print(X_test)

# --------Naive bayes------------#

# print(Y_test)
# print("---------------------")
clf = MultinomialNB()
clf.fit(X_train, Y_train)
Y_prediction=clf.predict(X_test)
# print(Y_prediction)
# confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_test, Y_prediction).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)
F1_score = 2 * (precision * recall) / (precision + recall)
print("Naive bayes prediction(unigram)")
print(tn, fp, fn, tp,"precision:",precision,"recall:",recall,"accuracy:",accuracy,"F1_score:",F1_score)
print("---------------------")

# --------Regularized logistic regression------------#
clf =  LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, Y_train)
Y_prediction=clf.predict(X_test)
# print(Y_prediction)
# confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_test, Y_prediction).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)
F1_score = 2 * (precision * recall) / (precision + recall)
print("Regularized logistic regression prediction(unigram)")
print(tn, fp, fn, tp,"precision:",precision,"recall:",recall,"accuracy:",accuracy,"F1_score:",F1_score)
print("---------------------")

# --------Classification trees------------ #

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
Y_prediction=clf.predict(X_test)
# print(Y_prediction)
# confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_test, Y_prediction).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)
F1_score = 2 * (precision * recall) / (precision + recall)
print("Classification trees prediction(unigram)")
print(tn, fp, fn, tp,"precision:",precision,"recall:",recall,"accuracy:",accuracy,"F1_score:",F1_score)
print("---------------------")

# --------Random Forest------------ #
n_estimators=100 # the number of trees
max_depth=None
min_samples_split=2  # The minimum number of samples required to split an internal node (default=2)
min_samples_leaf=1 # The minimum number of samples required to be at a leaf node

clf = RandomForestClassifier(n_estimators=n_estimators,max_features="sqrt")
clf = clf.fit(X_train, Y_train)
Y_prediction=clf.predict(X_test)
# print(Y_prediction)
# confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_test, Y_prediction).ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)
F1_score = 2 * (precision * recall) / (precision + recall)
print("Random Forest prediction(unigram)")
feature_importances=clf.feature_importances_
tempDic={}
tempList=[]
for i in bagOfWords:
    tempList.append(i)
for n in tempList:
    tempDic[n]=feature_importances[tempList.index(n)]
print(tn, fp, fn, tp,"precision:",precision,"recall:",recall,"accuracy:",accuracy,"F1_score:",F1_score)
print("---------------------")
# print(feature_importances)
# print(sorted(tempDic.items(), key=lambda k: k[1], reverse=True))