'''
Created on Sep 22, 2016

@author: DAMIEN.THIOULOUSE
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils
import re
import nltk
import numpy as np
from dask.dataframe.core import DataFrame
import Models
import SentimentScore as SS
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error
import sklearn


def ReviewToWords(s,stopwords):
    s = str(s)
    s = ''.join([i for i in s if not i.isdigit()])
    s = s.lower() # downcase
    s = s.encode('ascii','ignore')
    s = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',s)
    
    s = re.sub('[?\@#$,*./]','',s)
    s = re.sub('[\s]+', ' ', s) # remove unnessary spoaces
    s = s.strip('[\' "]')

    tokens = nltk.tokenize.word_tokenize(s)
    myTok = []
    if np.shape(tokens)[0] > 1:
        for term in tokens:
            if len(term) > 2:
                term = wordnet_lemmatizer.lemmatize(term)
                term = term.strip('[]\]')
                
                if term not in stopwords:
                    #mood_score += scores.get_score(term)
                    myTok.append(term)
        
        
#     tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
#     tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
#     tokens = [t.strip('[]\]') for t in tokens] # put words into base form
#     tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return myTok#,mood_score

def createWordIndex(chats):
    word_to_index = {}
    index_to_word = {}
    current_index = 0

#     positive_mood = []
#     negative_mood = []
    
    for tokens in chats:
        #tokens is a list of words
        #tokens = ReviewToWords(review.text,stopwords)
        #positive_tokenized.append(tokens)
        #positive_mood.append(posMood)
        
        
        for token in tokens:
            if token not in word_to_index:
                word_to_index[token] = current_index
                index_to_word[current_index] = token
                current_index += 1 

                
    return word_to_index,index_to_word#,positive_mood,negative_mood

def normalizematrix(data):
    row_sums = data.sum(axis = 0)
    data = data / row_sums[np.newaxis,: ]
    return data

    
def buildFeatures(chat,scores,PosNegSplit):
    woAgent = chat[chat.AGENT_ID == 0]
    woAgent['CLASS'] =  woAgent['SCORE'].apply(lambda x: 0 if x < PosNegSplit else 1 )
    woAgent['Tokens'] = woAgent['TEXT'].apply(lambda x: ReviewToWords(x,stopwords) )

    woAgent1 = woAgent.groupby('SESSION_ID')#,as_index = False)
    Tokens = woAgent1['Tokens'].sum() #list of words for each chat session
    Class = woAgent1['CLASS'].first()
    Score = woAgent1['SCORE'].first()
    NPS_LEVEL = woAgent1['NPS_LEVEL'].first()
    Count = woAgent1['CLASS'].count().rename("Count")
    max_per_group = woAgent1['TIMESTAMP'].max()
    min_per_group = woAgent1['TIMESTAMP'].min()
    duration = (max_per_group - min_per_group).apply(lambda x: x.seconds)
    
    
    
    
   # aggregated_df = DataFrame ()
    
    word_to_index,index_to_word = createWordIndex(Tokens)
    N = Count.size
    Mood = np.zeros(N)
    data = np.zeros((N, len(word_to_index) ))
    i=0
    for List in Tokens:
        xy, mood = tokens_to_vector(List,word_to_index,index_to_word,scores)
        data[i,:] = xy
        Mood[i] = mood
        i += 1
    
    mydf = pd.concat([Count,duration],axis = 1)
    mydf['Mood'] = Mood
    w = mydf.apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    #w = mydf.apply(lambda x: (x / x.max()))
    w['Class'] = Class  
    w['SCORE'] = Score
    w['NPS_LEVEL'] = NPS_LEVEL
   # w['SESSION_ID'] = w.index
    
    data = normalizematrix(data)
    #Mood = Mood/abs(Mood).max()
    #Count = Count/Count.max()
    t = w.reset_index()
 
    
    return data, t
    
def tokens_to_vector(tokens,word_to_index,index_to_word,scores):
    x = np.zeros(len(word_to_index)) # last element is for the label
    mood_score = 0
    for t in tokens:
        mood_score += scores.get_score(t)
        i = word_to_index[t]
        x[i] += 1
    
   # x = x / x.sum() # normalize it before setting label
    #x[-1] = label
    return x,mood_score

def splittrainTest(data,t,testSize):
    N = np.shape(data)[0]
    reorder = np.random.permutation(N)
    t = t.ix[reorder]
    data = data[reorder,]
    Ntest = int(N*testSize)
    Xtrain = data[:-Ntest,]
    Ytrain = t['Class'].iloc[:-Ntest]
    Xtest = data[-Ntest:,]
    Ytest =t['Class'].iloc[-Ntest:]
    Scoretest = t['SCORE'].iloc[-Ntest:]
    Scoretrain = t['SCORE'].iloc[:-Ntest]
    
    t.drop(['SESSION_ID','Class','SCORE'], axis=1, inplace=True)
    featurestest = t.iloc[-Ntest:]
    featurestrain = t.iloc[:-Ntest]
    
    return Xtrain , Ytrain, Xtest, Ytest, featurestest,featurestrain, Scoretest, Scoretrain

if __name__ == '__main__':
    
    scores = SS.SentimentScore(open("/Users/DAMIEN.THIOULOUSE/Documents/SentimentAnalysis/Datasets/AFINN-111.txt"))
    chat_df,stopwords = utils.parseChat("C:/Users/DAMIEN.THIOULOUSE/Documents/SentimentAnalysis/Datasets/Chat.txt")
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    data, t = buildFeatures(chat_df,scores, 7)
    
    t.drop(['TIMESTAMP','Count','NPS_LEVEL'], axis=1, inplace=True)
    #t.drop(['NPS_LEVEL'], axis=1, inplace=True)
    Xtrain , Ytrain, Xtest, Ytest, featurestest,featurestrain, Scoretest, Scoretrain = splittrainTest(data, t, 0.15)
    
    
    #myMod = Models.StackedModel(LogisticRegression(),svm.SVC(probability=True))
    #myMod.fit(Xtrain,featurestrain,Ytrain)
    myMod = Models.StackedModel(LogisticRegression(),LinearRegression())#DecisionTreeRegressor())
    myMod.fit(Xtrain,featurestrain,Ytrain,Scoretrain)
    OutSample = myMod.predict(Xtest,featurestest,Scoretest)
    
    CatPred = ['LOW' if x <= 6 else 'HIGH' if x >= 9 else 'MID' for x in OutSample]
    Cattrue = ['LOW' if x <= 6 else 'HIGH' if x >= 9 else 'MID' for x in Scoretest]
    print zip(Scoretest,OutSample)
    print zip(Cattrue,CatPred)
    #InSample = myMod.predict(Xtrain,featurestrain)
    print mean_squared_error(Scoretest,OutSample)
    Success_rate = sum([1  if x==y else 0 for x,y in zip(Cattrue,CatPred)])/float(np.shape(Cattrue)[0])
    print Success_rate
   # Models.plotAUC(Ytest,OutSample )
    #Models.plotAUC(Ytrain,InSample )
    
    
    #word_index_map,positive_tokenized,negative_tokenized  = getWordsVariables(chat_df,stopwords)