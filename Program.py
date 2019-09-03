#!/usr/bin/env python
# coding: utf-8

# In[494]:


#import required libraries
import numpy as np
import scipy as sp
import pandas as pd
import re
import string
import nltk
from collections import Counter
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time


# In[458]:


#test = pd.read_csv('255_test.dat', sep='\n',names=['description'])
Start = time.time
train = pd.read_csv("255_train.dat", 
                 sep="\n", #separator whitespace
                 header=None) 


# In[459]:


#train = pd.read_csv('255_train.dat', sep='\t', names=['class', 'description']) 
test = pd.read_csv("255_test.dat", 
                  sep="\n", #separator whitespace
                 header=None) 
val_test = test.iloc[:].values
val_train = train.iloc[:].values
clas_train =[]
desc_train =[]
desc_test =[]


# In[460]:


print("------------------------------------------------------")
print("Make list of descrption and Class")
for i in val_train:
    clas_train.append(i[0][:1])
    desc_train.append(i[0][2:])

for i in val_test:
    desc_test.append(i[0][0:])

print("The Description of training data is")


# In[465]:


clas_train = list(map(int, clas_train))


# In[401]:


def clean_punc(text):
    sentence_punc = []
    for sentence in text:
        string_punc =[]
        for word in sentence.split():
            word = re.sub(r'[?|!|\'|#|.]', r'', word)
            string_punc.append(word)
        string_punc=" ".join(string_punc) 
        sentence_punc.append(string_punc)
    return(sentence_punc)


# In[402]:


def lower(text):
    sentence_lower = []
    for sentence in text:
        string_low =[]
        for word in sentence.split():
            string_low.append(word.lower())
        string_low=" ".join(string_low)    
        sentence_lower.append(string_low)
    return(sentence_lower)
        


# In[403]:


def remove_stopwords(text):
   nltk.download('stopwords')
   from nltk.corpus import stopwords
   from nltk.stem.snowball import SnowballStemmer
   from nltk.stem.wordnet import WordNetLemmatizer
   stop = stopwords.words('english')
   sno = SnowballStemmer('english')
   sentence_stop = []
   for sentence in text:
       string_stop = []
       for word in sentence.split():
           if(word not in stop):
               string_stop.append(word)
       string_stop=" ".join(string_stop)
       sentence_stop.append(string_stop)
   return(sentence_stop)


def remove_short_words(text):
    sentence_short = []
    for sentence in text:
        string_short =[]
        for word in sentence.split():
            if (len(word) > 3):  
                string_short.append(word)
        string_short=" ".join(string_short)    
        sentence_short.append(string_short)
        #print(sentence_short)
    return(sentence_short)


# In[404]:


print("Remove Stopwords from training")
desc_train= remove_stopwords(desc_train)
print("Remove Stopwords from testing")
desc_test= remove_stopwords(desc_test)


# In[405]:


print("make the text lower case in training data")
desc_train = lower(desc_train)
print("make the text lower case in training data")
desc_test = lower(desc_test)


# In[406]:


print("Clean Punctuation in train")
desc_train = clean_punc(desc_train)
print("Clean Punctuation in test")
desc_test = clean_punc(desc_test)


print("Remove Minimum Length words in train")
desc_train = remove_short_words(desc_train)
print("Remove Minimum Length words in test")
desc_test = remove_short_words(desc_test)
# In[ ]:





# In[413]:

#Replace the null values with 0
sparse_dataframe = pd.DataFrame()
desc_test = pd.SparseDataFrame(desc_test).fillna(0)
desc_train = pd.SparseDataFrame(desc_train).fillna(0)
sparse_dataframe['text'] = pd.concat([desc_train[0], desc_test[0]], ignore_index=True, sort=False)


# In[421]:

# Create sparse matrix from the description lists
sparse_dataframe['text'] = pd.concat([desc_train, desc_test], ignore_index=True, sort=False)
sparse_dataframe_bow_transformer = CountVectorizer().fit(sparse_dataframe['text'])
sparse_dataframe_bow = CountVectorizer().fit(sparse_dataframe['text']).transform(sparse_dataframe['text'])
# Normalize the sparse matrix
train_tfidf = TfidfTransformer().fit(sparse_dataframe_bow).transform(sparse_dataframe_bow)


# In[496]:

#Calulate cosine similarity of test and train description
def getCosineSimilarity(train_tfidf):
    #for i in range(len(mat5_train)):
      #den1 = math.sqrt(i * i)
        #for j in range(len(mat5_train)):
    #    den2 = math.sqrt(j * j)
    A = pd.SparseDataFrame(train_tfidf).fillna(0)
    similarity = np.dot(A, A.T)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    #return temp3
    return cosine


# In[448]:

# Get the K nearest neighbors based on the cosine similarity, K and min-epsilon 
def getNearestNeighbors(X_test,y_train,k):
    output_tup = {}
    min_epsilon = 0.1
    for i in range(len(X_test)):
        final_votes = []
        class_index = []
        y_class = []
        bias = 0
        votes =  sorted(X_test[i],reverse = True)[:k]
        for a in votes:
            if a > 0.99:
                final_votes.append(a)
                bias = 1
		break
            elif a > min_epsilon:
                final_votes.append(a)
        if bias == 1:
            class_index.append(X_test[i].tolist().index(max(final_votes)))
        elif (len(final_votes) != 0):
            for p in range(len(final_votes)):
                class_index.append(X_test[i].tolist().index(votes[p]))
        else:
            class_index.append(X_test[i].tolist().index(max(votes)))
        for r in class_index:
            y_class.append(y_train[r])
        for m in y_class:
            output_tup[i] = Counter(y_class).most_common(1)[0][0]
    #print(output_tup)
    print("Length of output tup is")
    return output_tup


# In[487]:
# Execute the KNN algorithm and write an optput file with class predictions for the test data.
def KNNalgo_Prediction(train_tfidf,k):
   # for i in mat3:
   #     for j in mat5:
   #         Similarity_arr.append(getCosineSimilarity(i,j,clas))
    similarity_mat=getCosineSimilarity(train_tfidf)
    print("Similarity Matrix")
    print(similarity_mat)
    #Distribute_factor = 0.8 * 
    X_train = similarity_mat[:12000,:12000]
    y_train = clas_train[:12000]
    X_test = similarity_mat[12001:14438,:12000]
    y_test = clas_train[12001:14438]
    print("Make predictions for test sample")
    output = getNearestNeighbors(X_test,y_train,k)
    output_pred = list(output.values())
    print(classification_report(output_pred,y_test))
    print("start predicting the test datafile")
    X1_test = similarity_mat[14439:,:14438]
    #X1_train = similarity_mat[:14438,:14438]
    Y1_train = clas_train[:14438]
    Find_output = getNearestNeighbors(X1_test,Y1_train,k)
    Find_output_Values = list(Find_output.values())
    print("Write the output file")
    with open('255_pro1_output_7thMar_v2.txt', 'w') as f:
        for item in Find_output_Values:
            f.write("%s\n" % item)
    return 


# In[484]:


KNNalgo_Prediction(train_tfidf,34)
End = time.time
print("Time taken by the program")
#print(End - Start)

