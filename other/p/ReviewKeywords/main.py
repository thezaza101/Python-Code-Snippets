import nltk
import re
import os

data_path =  os.path.abspath(os.path.join('other','p','ReviewKeywords', 'datasets'))

def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

with open (data_path+"\\data.txt", "r",encoding="utf-8") as myfile:
    data=myfile.read()

data = data.lower()

from nltk.corpus import stopwords
stops = stopwords.words("English")

data = removeStopWords(stops, data)

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

fdist = FreqDist(word.lower() for word in word_tokenize(data)) 
# print(fdist.most_common(100))

from nltk import ngrams

nggrams = ngrams(data.split(), 3)

#for grams in nggrams:
#    print(grams)

fdist1 = nltk.FreqDist(nggrams)
for k,v in fdist1.most_common(100):
    print(k,v)