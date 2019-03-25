import io
import os
import re
import operator as op
from collections import Counter
import numpy as np
import pandas as pd

data_path =  os.path.abspath(os.path.join('other','p','MCNLPClass', 'datasets'))
file_path = data_path+'\\data.csv'
common_eng_words = []

def load_common_word_list(input_filename:str):
    input_file = open(input_filename, 'r',encoding="utf-8")
    file_contents = input_file.read()
    input_file.close()
    return file_contents.split('\n')
    
def int_filter( someList ):
    for v in someList:
        try:
            int(v)
            continue
        except ValueError:
            yield v

def unique_word_list(input_filename:str, writeFile:bool=False, output_filename:str=''):
    input_file = open(input_filename, 'r',encoding="utf8")
    file_contents = input_file.read().lower()
    input_file.close()
    word_list = re.split(r'(,+|\t+|\s+|;+|\|+|\n+)', file_contents)
    word_list1 = [w.replace(' ', '').replace(',', '').replace('0', '').replace('\n', '').replace('"', '').replace('\'', '') for w in word_list if len(w) >= 4]
    word_list2 = list(filter(None,int_filter(word_list1)))
    unique_words = list(set(word_list2))
    unique_words.sort()
    countList = Counter(word_list2)
    if writeFile:
        if output_filename=='':
            output_filename = 'unique_'+output_filename
        file = open(output_filename, 'w')
        for word in unique_words:
            file.write(str(word) + "\n")
        file.close()
    
    return unique_words, countList

def NormaliseDict(dictToNorm):
    maxVal = dictToNorm[max(dictToNorm.items(),key=op.itemgetter(1))[0]]
    minVal = dictToNorm[min(dictToNorm.items(),key=op.itemgetter(1))[0]]
    normWordList =	{"NA": 0}

    for w in dictToNorm:
        normWordList[w] = (dictToNorm[w] - minVal) / (maxVal - minVal) * 100
    
    return normWordList

def levenshtein(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

common_eng_words = load_common_word_list(data_path+"\\wiki-100k.txt")

unique_words_lst, count = unique_word_list(file_path, False)
count = NormaliseDict(count)


'''
for item in unique_words_lst:
    print("%s : %s" % (item,count[item]))


lst = sorted(count.items(),key = lambda kv: kv[1])
for item in lst:
    print("%s : %s" % (item[0],item[1]))
'''
listLen = len(unique_words_lst)
List1 = unique_words_lst
List2 = unique_words_lst

distMatrix = np.zeros((listLen,listLen),dtype=np.int)



for i in range(0,len(List1)):
  for j in range(i,len(List2)):
      distMatrix[i,j] = levenshtein(List1[i],List2[j])

outdf = pd.DataFrame(distMatrix)
outdf.columns = unique_words_lst
outdf['######'] = unique_words_lst
outdf = outdf.set_index('######')
#outdf.to_csv(data_path+'/dataOut.csv')

countMinVal = count[min(count.items(),key=op.itemgetter(1))[0]]


for w in count:
    if (count[w] == countMinVal and w.lower() not in common_eng_words):
        try:
            x = outdf.loc[w].to_dict()
            x = {k:v for k,v in x.items() if (v >= 1 and v <= 2 and k[0].lower() == w[0].lower())}
            if (len(x) > 0):
                correctionList = {k: count[k] for k in list(x.keys())}
                if(len(x)>1):
                    correctionList = {k: count[k] for k in x.keys()}
                #print(w)

                outDict = {k: (correctionList[k]+1)/((x[k]**x[k])**2) for k in correctionList.keys()}
                #outDict = sorted(outDict.items(), key=operator.itemgetter(1))
                print("%s : %s" % (w, outDict))

                #print(x)
                #print(correctionList)
                #print(outDict)
        except:
            print ("error...")
