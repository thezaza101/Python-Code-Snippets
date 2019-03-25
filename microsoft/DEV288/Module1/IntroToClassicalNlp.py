import nltk
import re

# nltk.download()

# 1.2 Crash Course in Regular Expressions 

'''
Operator   Meaning       Example  Example meaning

+          one or more   a+       look for 1 or more "a" characters 
*          zero or more  a*       look for 0 or more "a" characters
?          optional      a?       look for 0 or 1 "a" characters
[]         choose 1      [abc]    look for "a" or "b" or "c"
[-]        range         [a-z]    look for any character between "a" and "z"
[^]        not           [^a]     look for character that is not "a"
()         grouping      (a-z)+   look for one of more occurences of chars between "a" and "z"
(|)        or operator   (ey|ax)  look for strings "ey" or "ax"

ab         follow        ab       look for character "a" followed by character "b"
^          start         ^a       look for character "a" at start of string/line
$          end           a$       look for character "a" at end of string/line
\s         whitespace    \sa      look for whitespace character followed by "a"
.          any character a.b      look for "a" followed by any char followed by "b"
'''

# search for single char
print(re.search(r"x", "this is an extra helping"))

# search for single char
print(re.search(r"x", "this is an extra helping").group(0))     # gives easier-to-read output

# find all occurences of any character between "a" and "z"
print(re.findall(r"[a-z]", "$34.33 cash."))

# find all occurences of either "name:" or "phone:"
print(re.findall(r"(name|phone):", "My name: Joe, my phone: (312)555-1212"))

# find "lion", "lions" or "Lion", or "Lions"
print(re.findall(r"([Ll]ion)s?", "Give it to the Lions or the lion."))

# replace allll lowercase letters with "x"
print(re.sub("[a-z]", "x", "Hey.  I know this regex stuff..."))

# 2. Text processing

# shows how to access one of the gutenberg books included in NLTK
print("gutenberg book ids=", nltk.corpus.gutenberg.fileids())

# load words from "Alice in Wonderland"
alice = nltk.corpus.gutenberg.words("carroll-alice.txt")
print("len(alice)=", len(alice))
#print(alice[:100])

# load words from "Monty Python and the Holy Grail"
grail = nltk.corpus.webtext.words("grail.txt")
print("len(grail)=", len(grail))
#print(grail[:100])


'''** 2.2 Plain Text Extraction **

If your text data lives in a non-plain text file (WORD, POWERPOINT, PDF, HTML, etc.), you will need to use a “filter” to extract the plain text from the file.

Python has a number of libraries to extract plain text from popular file formats, but they are take searching and supporting code to use.
'''


'''
** 2.3 Word and Sentence Segmentation (Tokenization) **

Word Segmentation Issues:

    Some languages don’t white space characters
    Words with hyphens or apostrophes (Who’s at the drive-in?)
    Numbers, currency, percentages, dates, times (04/01/2018, $55,000.00)
    Ellipses, special characters

Sentence Segmentation Issues:

    Quoted speech within a sentence
    Abbreviations with periods (The Ph.D. was D.O.A)

Tokenization Techniques

    Perl script (50 lines) with RegEx (Grefenstette, 1999)

    maxmatch Algorithm:

    themanranafterit  ->    the man ran after it  
    thetabledownthere ->    theta bled own there      (Palmer, 2000)  

'''

# code example: simple version of maxmatch algorithm for tokenization (word segmentation)
def tokenize(str, dict):
    s = 0
    words = []
    
    while (s < len(str)):
        found = False
        
        # find biggest word in dict that matches str[s:xxx]
        for word in dict:
            lw = len(word)
            if (str[s:s+lw] == word):
                words.append(word)
                s += lw
                found = True
                break
        if (not found):
            words.append(str[s])
            s += 1

    print(words)
    #return words

# small dictionary of known words, longest words first
dict = ["before", "table", "theta", "after", "where", "there", "bled", "said", "lead", "man", "her", "own", "the", "ran", "it"]

# this algorithm is designed to work with languages that don't have whitespace characters
# so simulate that in our test
tokenize("themanranafterit", dict)      # works!
tokenize("thetabledownthere", dict)     # fails!

# NLTK example: WORD segmentation
print(nltk.word_tokenize("the man, he ran after it's $3.23 dog on 03/23/2016."))

# NLTK example: SENTENCE segmentation
print(nltk.sent_tokenize('The man ran after it.  The table down there?  Yes, down there!'))

# ** 2.4 Stopword Removal **

'''
Stopwords are common words that are "not interesting" for the app/task at hand.

Easy part – removing words that appear in list.
Tricky part – what to use for stop words? App-dependent. Standard lists, high-frequency words in your text, …
'''

# code example: simple algorithm for removing stopwords
stoppers = "a is of the this".split()

def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

print(removeStopWords(stoppers, "this is a test of the stop word removal code."))

# NLTK example: removing stopwords
from nltk.corpus import stopwords
stops = stopwords.words("English")

print("len(stops)=", len(stops))

print(removeStopWords(stops, "this is a test of the stop word removal code."))


# ** 2.5 Case Removal **
'''
Case removal is part of a larger task called Text Normalization, which includes:

    case removal
    stemming (covered in next section)

Goal of Case removal – converting all text to, for example, lower case
'''

# code example: case removal
str = 'The man ran after it.  The table down there?  Yes, down there!'
str.lower()

# ** 2.6 Stemming **

'''
Goal of Stemming: – stripping off endings and other pieces, called AFFIXES – for English, this is prefixes and suffixes.

- convert word to its base word, called the LEMMA / STEM (e.g., foxes -> fox)

Porter Stemmer

    100+ cascading “rewrite” rules
    ational -> ate (e.g., relational -> relate)
    ing -> (e.g., playing -> play)
    sess -> ss (e.g., grasses -> grass)
'''


# NLTK example: stemming

def stem_with_porter(words):
    porter = nltk.PorterStemmer()
    new_words = [porter.stem(w) for w in words]
    return new_words
    
def stem_with_lancaster(words):
    porter = nltk.LancasterStemmer()
    new_words = [porter.stem(w) for w in words]
    return new_words    
    
str = "Please don't unbuckle your seat-belt while I am driving, he said"

print("porter:", stem_with_porter(str.split()))
print()
print("lancaster:", stem_with_lancaster(str.split()))



# 3. Text Exploration

'''
** 3.1 Frequency Analysis **

    Frequency Analysis
    Letter
    Word
    Bigrams
    Plots

'''


# NLTK example: frequence analysis
import nltk
from nltk.corpus import gutenberg
from nltk.probability import FreqDist

# get raw text from "Sense and Sensibility" by Jane Austen
raw = gutenberg.raw("austen-sense.txt")
fd_letters = FreqDist(raw)

words = gutenberg.words("austen-sense.txt")
fd_words = FreqDist(words)
sas = nltk.Text(words)

# these 2 lines let us size the freq dist plot
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5)) 

# frequency plot for letters from SAS
fd_letters.plot(100)


# these 2 lines let us size the freq dist plot
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5)) 

# frequency plot for words from SAS
fd_words.plot(50)

