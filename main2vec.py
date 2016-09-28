'''
Created on Sep 26, 2016

@author: DAMIEN.THIOULOUSE
'''
from sklearn.manifold import TSNE
from gensim.models.word2vec import Word2Vec
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import utils
import re
import nltk
import SentimentScore as SS
from NNet import NeuralNet
from sklearn.metrics import mean_squared_error
import codecs
from nltk import pos_tag
from sklearn import neighbors
from sklearn.ensemble import BaggingRegressor
import Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#from NNet import NeuralNet
def ReviewToWords(s,stopwords):
    s = str(s)
    s = s.lower()
    myTok = []
    if "one of our tech coaches" not in s:
        s = ''.join([i for i in s if not i.isdigit()])
         # downcase
        #s = s.encode('ascii','ignore')
        s = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',s)
        s = re.sub('[\*/]','',s)
        #s = re.sub('[?\@#$,*./]','',s)
        #s = re.sub('[\s]+', ' ', s) # remove unnessary spoaces
       # s = s.strip('[\' "]')
       
    
    
        tokens = nltk.tokenize.word_tokenize(s)
        
        #sent = pos_tag(tokens)
        
        
        if np.shape(tokens)[0] >= 3:
            for term in tokens:
                #cat = term[1]
                w = term
                if len(w) >= 3:
                    w = wordnet_lemmatizer.lemmatize(w)
                    w = w.strip('[]\]')
                    
                    if w not in stopwords:
                        #mood_score += scores.get_score(term)
                        myTok.append(w)
        
#         sent = pos_tag(tokens)
#         
#         
#         if np.shape(tokens)[0] >= 3:
#             for term in sent:
#                 cat = term[1]
#                 w = term[0]
#                 if len(w) >= 3 and cat != 'NNP':
#                     w = wordnet_lemmatizer.lemmatize(w)
#                     w = w.strip('[]\]')
#                     
#                     if w not in stopwords:
#                         #mood_score += scores.get_score(term)
#                         myTok.append(w)
#             
            
    #     tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    #     tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    #     tokens = [t.strip('[]\]') for t in tokens] # put words into base form
    #     tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return myTok#,mood_score
  
def buildFeatures(chat,scores,PosNegSplit):
    
    woAgent = chat[chat.AGENT_ID == 0]
    #woAgent= chat
    #woAgent['CLASS'] =  woAgent['SCORE']
    woAgent['Tokens'] = woAgent['TEXT'].apply(lambda x: ReviewToWords(x,stopwords) )

    woAgent1 = woAgent.groupby('SESSION_ID')#,as_index = False)
    Tokens = woAgent1['Tokens'].sum() #list of words for each chat session
    #y = woAgent1['CLASS'].first()
    
    Score = woAgent1['SCORE'].first()
    
    y = Score.apply(lambda x: 0 if x < PosNegSplit else 1 ) #creates class label for potential classification
    #NPS_LEVEL = woAgent1['NPS_LEVEL'].first()
#     Count = woAgent1['CLASS'].count().rename("Count")
#     max_per_group = woAgent1['TIMESTAMP'].max()
#     min_per_group = woAgent1['TIMESTAMP'].min()
#     duration = (max_per_group - min_per_group).apply(lambda x: x.seconds)
#     
#     
#     
#     
#    # aggregated_df = DataFrame ()
#     
#     word_to_index,index_to_word = createWordIndex(Tokens)
#     N = Count.size
#     Mood = np.zeros(N)
#     data = np.zeros((N, len(word_to_index) ))
#     i=0
#     for List in Tokens:
#         xy, mood = tokens_to_vector(List,word_to_index,index_to_word,scores)
#         data[i,:] = xy
#         Mood[i] = mood
#         i += 1
#     
#     mydf = pd.concat([Count,duration],axis = 1)
#     mydf['Mood'] = Mood
#     w = mydf.apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
#     #w = mydf.apply(lambda x: (x / x.max()))
#     w['Class'] = Class  
#     w['SCORE'] = Score
#     w['NPS_LEVEL'] = NPS_LEVEL
#     w['Tokens'] = Tokens
#    # w['SESSION_ID'] = w.index
#     
#     data = normalizematrix(data)
#     #Mood = Mood/abs(Mood).max()
#     #Count = Count/Count.max()
#     t = w.reset_index()
 
    
    return np.vstack((y.values,Score.values)).T,Tokens

#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def getCoefWords(word,size):
    w = str(word)
    try:
        return imdb_w2v[w].reshape((1, size))
        
    except KeyError:
        pass


def plot():
 
    embeddings_file = 'w2c.txt'
    wv, vocabulary = load_embeddings(embeddings_file)
 
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:1000,:])
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
 
 
def load_embeddings(file_name):
 
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        f_in.next()
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in])

    wv = np.loadtxt(wv)
    return wv, vocabulary


    

if __name__ == '__main__':
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    chat_df,stopwords = utils.parseChat("C:/Users/DAMIEN.THIOULOUSE/Documents/SentimentAnalysis/Datasets/Chat.txt")
    scores = SS.SentimentScore(open("/Users/DAMIEN.THIOULOUSE/Documents/SentimentAnalysis/Datasets/AFINN-111.txt"))
    y,x = buildFeatures(chat_df,scores, 8)
   
    #y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    n_dim = 50
    #Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=5)
    imdb_w2v.build_vocab(x_train)
    imdb_w2v.train(x_train)
    imdb_w2v.save_word2vec_format('w2c.txt', fvocab=None, binary=False)
    imdb_w2v.train(x_test)
    #imdb_w2v.save_word2vec_format('w2c.txt', fvocab=None, binary=False)
    #plot()
    #imdb_w2v.similar_by_word('battery',topn=50)
    #imdb_w2v.most_similar(positive=['screen', 'broken'], negative=['phone'])
    #imdb_w2v.most_similar_cosmul(positive=['screen','drain'], negative=['battery'])
    
#     ts = TSNE(2)
#     red = np.concatenate([getCoefWords(z, n_dim) for z in np.concatenate(x_train)])
#     reduced_vecs = ts.fit_transform(red)
#     plt.plot(reduced_vecs[:100,0], reduced_vecs[:100,1], marker='o')
#     for i in range(len(reduced_vecs)):
#         plt.text(reduced_vecs[:100,0], reduced_vecs[:100,1], np.concatenate(x_train)[:100])  
#     
    
    #
    train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
    train_vecs = scale(train_vecs)
    
    #Train word2vec on test tweets
    #imdb_w2v.train(x_test)
    
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
    test_vecs = scale(test_vecs)
    
    #Bagging with KNN regressor
#     clf = BaggingRegressor(base_estimator=neighbors.KNeighborsRegressor(3),n_estimators=20,bootstrap=True,oob_score=True)
#     clf.fit(train_vecs, y_train[:,1])
#     knnRegInsampleBagg = clf.predict(train_vecs)
#     knnRegOutsampleBagg = clf.predict(test_vecs)   
    
    


################ Regression
    knn = neighbors.KNeighborsRegressor(4)
    knn.fit(train_vecs, y_train[:,1])
    knnRegInsample = knn.predict(train_vecs)
    knnRegOutsample = knn.predict(test_vecs)
    print "MSE 1-10 KNN"
    Models.printMSE(knnRegOutsample,y_test[:,1],knnRegInsample,y_train[:,1])
    NPS_SCORE_pred_out,NPS_SCORE_true_out, NPS_SCORE_pred_in,NPS_SCORE_true_in = Models.NPSfrom1_10(knnRegOutsample, y_test[:,1], knnRegInsample, y_train[:,1])
    print "KNN - MSE - NPS:"
    Models.printMSE(NPS_SCORE_pred_out,NPS_SCORE_true_out,NPS_SCORE_pred_in,NPS_SCORE_true_in)
    
    
    treshol_low = 0.2
    treshold_high = 0.7

    lr = RandomForestClassifier(max_depth=50,n_estimators=10)#SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train[:,0])
    SGDInsample = lr.predict_proba(train_vecs)[:,1]*10
    SGDOutsample = lr.predict_proba(test_vecs)[:,1]*10
    print "MSE 1-10 SGD"
    Models.printMSE(SGDOutsample,y_test[:,1],SGDInsample,y_train[:,1])
    NPS_SCORE_pred_out,NPS_SCORE_true_out, NPS_SCORE_pred_in,NPS_SCORE_true_in = Models.NPSfrom1_10(SGDOutsample, y_test[:,1], SGDInsample, y_train[:,1])
    print "SGD - MSE - NPS:"
    Models.printMSE(NPS_SCORE_pred_out,NPS_SCORE_true_out,NPS_SCORE_pred_in,NPS_SCORE_true_in)

    
    StackedInsample = np.mean([SGDInsample,knnRegInsample],axis=0)
    StackedOutsample = np.mean([SGDOutsample,knnRegOutsample],axis=0)
    print "MSE 1-10 (SGD,KNN) stacked"
    Models.printMSE(StackedOutsample,y_test[:,1],StackedInsample,y_train[:,1])
    NPS_SCORE_pred_out,NPS_SCORE_true_out, NPS_SCORE_pred_in,NPS_SCORE_true_in = Models.NPSfrom1_10(StackedOutsample, y_test[:,1], StackedInsample, y_train[:,1])
    print "(SGD,KNN) stacked - MSE - NPS:"
    Models.printMSE(NPS_SCORE_pred_out,NPS_SCORE_true_out,NPS_SCORE_pred_in,NPS_SCORE_true_in)
    
    
    nnet = NeuralNet(100, learn_rate=1e-1, penalty=1e-8)
    maxiter = 1000
    batch = 150
    _ = nnet.fit(train_vecs, y_train[:,0], fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)
    
    print 'Test Accuracy: %.2f'%nnet.score(test_vecs, y_test[:,0])
    
    

    #print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)
    pred_probas_out = nnet.predict(test_vecs)[:,1]
    pred_probas_in = nnet.predict(train_vecs)[:,1]
    
    
    
    #pred_probas = lr.predict_proba(test_vecs)[:,1]
    
    #CatPred = ['LOW' if x <= 0.3 else 'HIGH' if x >= 0.7 else 'MID' for x in pred_probas_out]
    #Cattrue = ['LOW' if x <= 6 else 'HIGH' if x >= 9 else 'MID' for x in y_test.values[:,1]] 
    
    treshol_low = 0.2
    treshold_high = 0.7

    NPS_SCORE_pred_out,NPS_SCORE_true_out, NPS_SCORE_pred_in,NPS_SCORE_true_in = Models.NPSfrom1_10(pred_probas_out*10, y_test[:,1], pred_probas_in*10, y_train[:,1])
    #Success_rate = sum([1  if x==y else 0 for x,y in zip(Cattrue,CatPred)])/float(np.shape(Cattrue)[0])

    print "MSE:"
    print "insample: ",mean_squared_error(NPS_SCORE_true_in,NPS_SCORE_pred_in)
    print "outsample: ",mean_squared_error(NPS_SCORE_true_out,NPS_SCORE_pred_out)

    
    Models.plotAUC(y_test[:,0], pred_probas_out)






#     NPS_SCORE_pred_out = [-100 if x <= 6 else 100 if x >= 9 else 0 for x in knnRegOutsampleBagg]
#     NPS_SCORE_true_out = [-100 if x <= 6 else 100 if x >= 9 else 0 for x in y_test[:,1]]
#     NPS_SCORE_pred_in = [-100 if x <= 6 else 100 if x >= 9 else 0 for x in knnRegInsampleBagg]
#     NPS_SCORE_true_in = [-100 if x <= 6 else 100 if x >= 9 else 0 for x in y_train[:,1]]    
    #Success_rate = sum([1  if x==y else 0 for x,y in zip(Cattrue,CatPred)])/float(np.shape(Cattrue)[0])
    
#     Success_rate_out = sum([1  if x==y else 0 for x,y in zip(NPS_SCORE_true_out,NPS_SCORE_pred_out)])/float(np.shape(NPS_SCORE_true_out)[0])
#     Success_rate_in = sum([1  if x==y else 0 for x,y in zip(NPS_SCORE_pred_in,NPS_SCORE_true_in)])/float(np.shape(NPS_SCORE_pred_in)[0])
