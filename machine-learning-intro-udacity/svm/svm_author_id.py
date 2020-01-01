#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
clf = SVC(kernel = "rbf", C = 10000) # Create the classifier with a linear kernel

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train, labels_train) # Train the classifier
print "training time: ", round(time()-t0, 3), "s" # 104.71 s

t1 = time()
pred = clf.predict(features_test) # Predict the test labels
print "predicting time: ", round(time()-t1, 3), "s" # 10.251 s

accuracy = accuracy_score(pred, labels_test)
print accuracy # 0.99	

chris_pred = []
for i in pred:
	if i == 1:
		chris_pred.append(i)
print len(chris_pred)

#########################################################


