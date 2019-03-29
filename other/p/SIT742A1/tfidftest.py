# https://nlpforhackers.io/tf-idf/

from nltk.corpus import reuters

print
reuters.fileids()  # The list of file names inside the corpus
print
len(reuters.fileids())  # Number of files in the corpus = 10788

# Print the categories associated with a file
print
reuters.categories('training/999')  # [u'interest', u'money-fx']

# Print the contents of the file
print
reuters.raw('test/14829')

from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize

stop_words = stopwords.words('english') + list(punctuation)


def tokenize(text):
	words = word_tokenize(text)
	words = [w.lower() for w in words]
	return [w for w in words if w not in stop_words and not w.isdigit()]


# build the vocabulary in one pass
vocabulary = set()
for file_id in reuters.fileids():
	words = tokenize(reuters.raw(file_id))
	vocabulary.update(words)

vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}

VOCABULARY_SIZE = len(vocabulary)
DOCUMENTS_COUNT = len(reuters.fileids())

print
VOCABULARY_SIZE, DOCUMENTS_COUNT  # 10788, 51581



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize, vocabulary=vocabulary)

# Fit the TfIdf model
tfidf.fit([reuters.raw(file_id) for file_id in reuters.fileids()])

# Transform a document into TfIdf coordinates
X = tfidf.transform([reuters.raw('test/14829')])

# Check out some frequencies
print
X[0, tfidf.vocabulary_['year']]  # 0.0562524229373
print
X[0, tfidf.vocabulary_['following']]  # 0.057140265658
print
X[0, tfidf.vocabulary_['provided']]  # 0.0689364372666
print
X[0, tfidf.vocabulary_['structural']]  # 0.0900802810906
print
X[0, tfidf.vocabulary_['japanese']]  # 0.114492409303
print
X[0, tfidf.vocabulary_['downtrend']]  # 0.111137191743