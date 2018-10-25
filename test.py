import nltk
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

def bigram(sentence):
    nltk_tokens = nltk.word_tokenize(sentence)
    myList = list(nltk.bigrams(nltk_tokens))
    Bigrams = []
    for i in myList:
        Bigrams.append((''.join([w + ' ' for w in i])).strip())

    return Bigrams


def sentenceProcessing(sentence):
    tempList = []
    stop_words = list(get_stop_words('en'))  # About 900 stopwords
    nltk_words = list(stopwords.words('english'))  # About 150 stopwords
    stop_words.extend(nltk_words)
    stop_words.append("th")
    for i in string.punctuation:
        sentence = sentence.replace(i, '')
    lowerCase = sentence.lower().split()
    #print(bigramSentence)
    for eachWord in lowerCase:
        if eachWord not in stop_words:
            tempList.append(eachWord)
    sentenceWithoutStopWords = ' '.join(tempList)
    bigramSentence = bigram(sentenceWithoutStopWords)
    # print(tempList)
    return bigramSentence

