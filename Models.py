'''
Created on Sep 23, 2016

@author: DAMIEN.THIOULOUSE
'''

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
#import neuralnetwork 
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from collections import Counter

import numpy as np

class StackedModel(object):

    def __init__(self, mod1 , mod2):
        self.model1 = mod1
        self.model2= mod2
        pass
    
    def fit(self, Words, features, Y1,Y2):
        #self.model1 = LogisticRegression()#GaussianNB()#svm.LinearSVC()#
        self.model1.fit(Words, Y1)
        Yint = self.model1.predict(Words)
        #Yint = self.model1.predict_proba(Words)[:,1]
        Xint = np.column_stack(( features.as_matrix(), Yint))
        #self.model2 = LogisticRegression()#GaussianNB()#svm.LinearSVC()#
        
        
        
        
        self.model2.fit(Xint, Y2)   
        
    def predict(self, Words, features,Scoretest):
        Yint = self.model1.predict(Words)
        Xint= np.column_stack(( features.as_matrix(), Yint))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(features.ix[:,0].values, Yint,Scoretest )
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        #Yfinal = self.model2.predict_proba(Xint)[:,1]
        Yfinal = self.model2.predict(Xint)
        
        
        return Yfinal

def logistic_regression_pred(Xtrain,Ytrain,Xtest,prob = True):
    model = LogisticRegression()#GaussianNB()#svm.LinearSVC()#
    model.fit(Xtrain, Ytrain)
    if prob == True:
        
        Ypred= model.predict_proba(Xtest)[:,1] 
    else:
        Ypred= model.predict(Xtest)
        
    return Ypred

def Bayesian_pred(Xtrain,Ytrain,Xtest):
    model = GaussianNB()#LogisticRegression()#GaussianNB()#svm.LinearSVC()#
    model.fit(Xtrain, Ytrain)
    return model.predict(Xtest)

def svm_pred(Xtrain,Ytrain,Xtest):
    model = GaussianNB()#LogisticRegression()#GaussianNB()#svm.LinearSVC()#
    model.fit(Xtrain, Ytrain)
    return model.predict(Xtest)

def myNNet_pred(Xtrain,Ytrain,Xtest):
    nn = neuralnetwork.NeuralNetwork.init(
    lambda_val = 0.03, # you know what lambda is, right? it is the regularization parameter
    input_layer_size = int(np.shape(Xtrain)[1]), # number of features in each input row
    output_layer_size = 1, # number of output classes (use 1 for binary)
    hidden_layer_sizes = [3]#[800,300] # array like structure, mentioning size of hidden layers
)
     
    model = nn.train(Xtrain, Ytrain)
    return model,model.predict_binary_classification(Xtest)

def classification_metrics(Y_pred, Y_true):
    #TODO: Calculate the above mentioned metrics
    #NOTE: It is important to provide the output in the same order
    acc= accuracy_score(Y_true, Y_pred)
    auc_= roc_auc_score(Y_true, Y_pred)
    precision= precision_score(Y_true, Y_pred)
    recall= recall_score(Y_true, Y_pred)
    f1score= f1_score(Y_true, Y_pred)
    
    return acc, auc_, precision, recall, f1score

def decisionTree_pred(X_train, Y_train, X_test):
    #TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
    #IMPORTANT: use max_depth as 5. Else your test cases might fail.
    mod = DecisionTreeClassifier(max_depth=5, random_state=545510477)
    mod1=mod.fit(X_train,Y_train)
    Y_pred=mod1.predict(X_test)
    return Y_pred

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
    print "______________________________________________"
    print "Classifier: "+classifierName
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
    print "Accuracy: "+str(acc)
    print "AUC: "+str(auc_)
    print "Precision: "+str(precision)
    print "Recall: "+str(recall)
    print "F1-score: "+str(f1score)
    print "______________________________________________"
    print ""
    

def NPSfrom1_10(outSamplePred,outSampleTrue,inSamplePred,inSampleTrue,low = 6,high=9):
    NPS_SCORE_pred_out = [-100 if x <= low else 100 if x >= high else 0 for x in outSamplePred]
    NPS_SCORE_true_out = [-100 if x <= low else 100 if x >= high else 0 for x in outSampleTrue]
    NPS_SCORE_pred_in = [-100 if x <= low else 100 if x >= high else 0 for x in inSamplePred]
    NPS_SCORE_true_in = [-100 if x <= low else 100 if x >= high else 0 for x in inSampleTrue]  
    return NPS_SCORE_pred_out,NPS_SCORE_true_out, NPS_SCORE_pred_in,NPS_SCORE_true_in     
    
def printMSE(outSamplePred,outSampleTrue,inSamplePred,inSampleTrue):
    print "insample: ",mean_squared_error(inSampleTrue,inSamplePred)
    print "outsample: ",mean_squared_error(outSampleTrue,outSamplePred)
    
def printCategories(NPS_SCORE_pred_out,NPS_SCORE_true_out):
    success = [1  if x==y else 0 for x,y in zip(NPS_SCORE_true_out,NPS_SCORE_pred_out)]
    
    high_true = [1  if x==100 else 0 for x in NPS_SCORE_true_out]
    mid_true = [1  if x==0 else 0 for x in NPS_SCORE_true_out]
    low_true = [1  if x==-100 else 0 for x in NPS_SCORE_true_out]
    
    Highpred = [NPS_SCORE_pred_out[i] for i in range(len(NPS_SCORE_pred_out)) if high_true[i] == 1]
    Midpred = [NPS_SCORE_pred_out[i] for i in range(len(NPS_SCORE_pred_out)) if mid_true[i] == 1]
    Lowpred = [NPS_SCORE_pred_out[i] for i in range(len(NPS_SCORE_pred_out)) if low_true[i] == 1]
    
    high = Counter(Highpred)
    mid = Counter(Midpred)
    low = Counter(Lowpred)
    
    true = Counter(NPS_SCORE_true_out)
    
    correctpred = [NPS_SCORE_pred_out[i] for i in range(len(NPS_SCORE_pred_out)) if success[i] == 1]
    predsuccess = Counter(correctpred)
    
    def get_num(a,b):
        return np.round(100*float(a)/float(b),2)
        
    
    print "Success rate"
    print "HIGH -", get_num(high.get(100),np.sum(high.values())),"%","(others in MID -",get_num(high.get(0),np.sum(high.values())),", LOW -",get_num(high.get(-100),np.sum(high.values())),")"
    print "MID - ", get_num(mid.get(0),np.sum(mid.values())),"%","(others in HIGH -",get_num(mid.get(100),np.sum(mid.values())),", LOW -",get_num(mid.get(-100),np.sum(mid.values())),")"
    print "LOW - ", get_num(low.get(-100),np.sum(low.values())),"%","(others in HIGH -",get_num(low.get(100),np.sum(low.values())),", MID -",get_num(low.get(0),np.sum(low.values())),")"
    
def averageAUC(data,Label):
    
    aucArray = []
    for i in range(30):
        reorder = np.random.permutation(np.shape(Label)[0])
        X = data[reorder,:]
        Y = Label[reorder]
   # reviews = zip(reorder, Allreviews)
    #reviews.sort()

        Ntest = 10
        Xtrain = X[:-Ntest,]
        Ytrain = Y[:-Ntest,]
        Xtest = X[-Ntest:,]
        Ytest = Y[-Ntest:,]
        model = LogisticRegression()#GaussianNB()#svm.LinearSVC()#
        model.fit(Xtrain, Ytrain)
        aucArray.append(roc_auc_score(Ytest, model.predict_proba(Xtest)[:,1]))
    print "average AUC: ", np.mean(aucArray)  
    
    
def plotAUC(Ytest,Ypred):
    fpr, tpr, _ = roc_curve(Ytest, Ypred)
    roc_auc = auc(fpr, tpr)
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()