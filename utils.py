'''
Created on Sep 22, 2016

@author: DAMIEN.THIOULOUSE
'''
import pandas as pd 
import numpy as np

def parseChat(filename):
    
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    dtypes={'AGENT_ID': np.int32, 'SESSION_ID': str,'SKILL_ID': np.int32}
    t = pd.read_csv(filename, sep='\t',dtype=dtypes, parse_dates=['TIMESTAMP'], date_parser=dateparse)#, nrows=1000 )
    
    stopwords = set(w.rstrip() for w in open("/Users/DAMIEN.THIOULOUSE/Documents/SentimentAnalysis/Datasets/stopwordsNEW.txt")) | {"URL"}
    
    return t,stopwords