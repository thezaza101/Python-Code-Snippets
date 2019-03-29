import pandas as pd

'''The first task is to read the json file as a Pandas DataFrame and delete the rows
which contain invalid values in the attributes of “points” and “price”.'''
df = pd.read_json('datasets//wine.json')
df = df.dropna(subset=['points', 'price'])

'''what are the 10 varieties of wine which receives the highest number of reviews?'''
dfTop10MostReviews = df['variety'].value_counts()[:10]

print("Q1:")
print(dfTop10MostReviews)
print('\n')
'''which varieties of wine having the average price less than 20, with the average points at least 90?'''
averagePoints = df.groupby('variety', as_index=False)['points'].mean()
averagePoints = averagePoints.loc[averagePoints['points']>=90]
averagePrice = df.groupby('variety', as_index=False)['price'].mean()
averagePrice = averagePrice.loc[averagePrice['price']<20]
q2 = pd.merge(averagePrice, averagePoints, on='variety')


print("Q2:")
print(q2)
print('\n')
'''
In addition, you need to group all reviews by different countries and generate a statistic
table, and save as a csv file named “statisticByState.csv”. The table must have four
columns:
Country – listing the unique country name.
Variety – listing the varieties receiving the most reviews in that country.
AvgPoint – listing the average point (rounded to 2 decimal places) of wine in that
country
AvgPrice – listing the average price (rounded to 2 decimal places) of wine in that country
'''

countryList = df['country'].drop_duplicates().to_frame()
dfTopReviews = df.groupby('country')['variety'].value_counts()
dfTopReviews = dfTopReviews.to_frame()
dfTopReviews.columns = ['Var_count']
dfTopReviews = dfTopReviews.reset_index(inplace=False)  
dfTopReviews = dfTopReviews.set_index(['country', 'variety'],drop=False, inplace=False)
dfTopReviews = dfTopReviews.drop_duplicates(subset='country', keep='first', inplace=False)

averagePointsCt = df.groupby('country', as_index=False)['points'].mean().round(2)
averagePriceCt = df.groupby('country', as_index=False)['price'].mean().round(2)

ss = pd.merge(countryList,dfTopReviews,on='country')
ss = pd.merge(ss,averagePointsCt,on='country')
ss = pd.merge(ss,averagePriceCt,on='country')
ss = ss[['country','variety','points','price']]
ss.to_csv('datasets//StatisticByStateSP.csv')

print("Q3:")
print("See 'datasets//StatisticByStateSP.csv' for more...")
print(ss)

'''In this task, you are required to write Python code to extract keywords from the
“description” column of the json data, used to redesign the wine menu for Hotel
TULIP.
You need to generate two txt files:'''


import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.probability import *
from itertools import chain
#from tqdm import tqdm
import codecs

with open('datasets//stopwords.txt') as f:
    stop_words = f.read().splitlines()
stop_words = set(stop_words)

'''HighFreq.txt This file contains the frequent unigrams that appear in more than 5000
reviews (one row in the dataframe is one review).'''

# write your code here
# define your tokenize

descData = df["description"]

def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

tokenizer = RegexpTokenizer(r"\w+(?:[-']\w+)?")

# remove stop words and tokenize each review
tokenized_Reviews = list((tokenizer.tokenize(removeStopWords(stop_words,review)) for review in descData))

# flatten the list of lists into a single list and also make everything lowercase
tokenized_words = [item.lower() for sublist in tokenized_Reviews for item in sublist]

# get the frequency distribution
fd = FreqDist(tokenized_words)

# select words with > 5000 frequency
fiveKOrMore = list(filter(lambda x: x[1]>5000,fd.items()))

# sort the list by the word
fiveKOrMore.sort(key=lambda tup: tup[0])

topCommonWords = list((word[0] for word in fiveKOrMore))

with open('datasets//HighFreq.txt', 'w') as f:
    for item in topCommonWords:
        f.write("%s\n" % item)

'''Shirazkey.txt This file contains the key unigrams with tf-idf score higher than 0:4.
To reduce the runtime, first you need to extract the description from the variety of
“Shiraz”, and then calculate tf-idf score for the unigrams in these descriptions
only.'''


# select 'description' from 'variety' eqaul to  'Shiraz'
descDataShiraz = df[df["variety"]=="Shiraz"]["description"]

# remove stop words and tokenize each review
tokenized_Reviews_Shiraz = list((tokenizer.tokenize(removeStopWords(stop_words,review.lower())) for review in descData))

idgen = (str(x) for x in range(0,len(tokenized_Reviews_Shiraz)))

doclist_Shiraz = {next(idgen):review for review in tokenized_Reviews_Shiraz}



# use TfidfVectorizer to calculate TF-IDF score
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', stop_words = 'english')

tfs = tfidf.fit_transform([' '.join(value) for value in doclist_Shiraz.values()])
print(tfs.shape)

# find words with TF-IDF score >0.4 and sort them
vocab = tfidf.get_feature_names()
tfidfScores = list(zip(vocab, tfs.toarray()[0]))

'''
temparry = tfs[:,0]
temparry = temparry.toarray()

for word, weight in zip(vocab, temparry):
    if weight > 0.4:
        print (word, ":", weight)
'''

# Print the list
for item in tfidfScores:
    if item[1] > 0.0:
        print (item[0], ":", item[1])

# save your table to 'key_Shiraz.txt'
with open('datasets//key_Shiraz.txt', 'w') as f:
    for item in tfidfScores:
        if item[1] > 0.4:
            f.write("%s\n" % item[0])