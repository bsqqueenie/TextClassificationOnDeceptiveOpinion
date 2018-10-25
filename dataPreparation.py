import pandas as pd
import numpy as np
def word2Vector(tokenizedText,allWords):
    vectorList=[]
    bagofWordList=list(allWords)
    print(bagofWordList)
    for i in range(len(allWords)):
        vectorList.append(0)
    for rowNum in range(tokenizedText.shape[0]):
        for eachWord in tokenizedText.iloc[rowNum,0]:
            if eachWord in bagofWordList:
                vectorList[bagofWordList.index(eachWord)]=1
        print(vectorList)
    tokenizedText.at[0, "text"] = vectorList
    return  tokenizedText



bagofWords={"i":1,"love":2,"you":3}
s="i love you"
s=s.split()

dic={"text":[s],"label":[1]}
new=pd.DataFrame(dic)
#print(new)
print(word2Vector(new,bagofWords))
# [1,1,1]
# new.at[0,"text"] = ['m', 'n']
# print(new)
