#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
clf = GaussianNB() # Create the classifier
t0 = time()
clf.fit(features_train, labels_train) # Train the classifier
print "training time: ", round(time()-t0, 3), "s" # 0.842 s

t1 = time()
pred = clf.predict(features_test) # Testing the classifier 
print "predicting time: ", round(time()-t1, 3), "s" # 0.219 s

accuracy = accuracy_score(pred, labels_test) # Accuracy of the classifier
print accuracy # 0.97

#########################################################


