import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np

fileName="d_hilton_1.txt"
openFile=open("d_hilton_1.txt","r")

#######
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems
########

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features =50
)

for i in openFile.readlines():
    text=[]
    text.append(i)
    corpus_data_features = vectorizer.fit_transform(text).toarray()
    vocab = vectorizer.get_feature_names()
    print(vocab)


#print(corpus_data_features)



#
# vocab = vectorizer.get_feature_names()
# print (vocab)