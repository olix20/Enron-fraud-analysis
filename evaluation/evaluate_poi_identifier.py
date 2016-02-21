#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here

from sklearn import tree
clf = tree.DecisionTreeClassifier()
# clf = clf.fit(features, labels)
# print clf.score(features,labels)



def findAccuracy(pred, actual,sign):

    total = len(pred[pred==sign])
    count = 0
    for i in range(0,len(pred)):
        if pred[i] ==sign and actual[i] ==sign:
            count+=1


    return float(count)/len(pred)

from sklearn.metrics import *
from sklearn import cross_validation


X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
clf = clf.fit(X_train, y_train)
print clf.score(X_test,y_test)
print "num people: ", len(y_test)
pred = clf.predict(X_test)
print "POIs predicted: ",len(pred[pred>0])
print "not POI accuracy: ", findAccuracy(pred,y_test,0)
print "POI accuracy: ", findAccuracy(pred,y_test,1)

print "precision: ", precision_score(y_test,pred)
print "recall: ", recall_score(y_test,pred)
