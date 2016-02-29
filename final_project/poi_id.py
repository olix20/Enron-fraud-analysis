#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn import tree

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', "total_stock_value","total_payments",
"shared_receipt_with_poi","restricted_stock","expenses",
"exercised_stock_options","total_correspondence_with_poi",
"bonus"]


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)

for name in data_dict:

    data_point = data_dict[name]
    if data_point["from_poi_to_this_person"]=="NaN" and \
    data_point["from_this_person_to_poi"]=="NaN":
        data_point["total_correspondence_with_poi"] = "NaN"
    else:
        data_point["total_correspondence_with_poi"] = \
            data_point["from_poi_to_this_person"]+\
                data_point["from_this_person_to_poi"]


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# original_features = features
#
# ### See how features compare to each other
# skb = SelectKBest(k=9)
# skb.fit(features,labels)
# print "SelectKBest scores for initial features: \n", skb.scores_
#
# ### Choose the highest ranking features (4)
#
# skb = SelectKBest(k=4)
# skb.fit(features,labels)
# features_selected_bool  =  skb.get_support()
# features_selected_list = [x for x, y in zip(features_list[1:],
# features_selected_bool ) if y]

## Filter dataset again to include final top features only
# data = featureFormat(my_dataset, ["poi"]+features_selected_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

ada_clf = AdaBoostClassifier(DecisionTreeClassifier())
ada_clf.fit(features_train,labels_train)
print "\nAdaboost accuracy:",ada_clf.score(features_test,labels_test)
print "\nAdaboost performance: \n",\
classification_report(labels_test, ada_clf.predict(features_test))
print "\nAdaboost model:\n", ada_clf




from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

knn_clf = KNeighborsClassifier()
scaled_features_train = MinMaxScaler().fit_transform(features_train)
scaled_features_test = MinMaxScaler().fit_transform(features_test)

knn_clf.fit(scaled_features_train,labels_train)

print "\nK-Nearest Neighbors accuracy:",\
knn_clf.score(scaled_features_test,labels_test)
print "\nK-Nearest Neighbors performance: \n",\
classification_report(labels_test, knn_clf.predict(scaled_features_test))
print "\nK-Nearest Neighbors model:\n", knn_clf



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html




### KNN Tuning
## KNN is sensetive to different features scales, so we'll normalize all
## features to (0,1) first

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

knn_pipeline = Pipeline([("scaler", MinMaxScaler()), ("skb", SelectKBest()),
("clf", KNeighborsClassifier())])


knn_param_grid = dict(skb__k=range(1, 4),clf__n_neighbors=[1,2,5,10],
clf__weights=['uniform','distance'], clf__algorithm=['ball_tree','kd_tree'],
clf__leaf_size=[1,2,5,10])

grid_search = GridSearchCV(knn_pipeline, param_grid=knn_param_grid,
scoring="recall",cv=5)

grid_search.fit(features_train, labels_train)
print "\n KNN best estimator: \n", (grid_search.best_estimator_),\
"\n best score:\n",grid_search.best_score_ ,\
"\n best params:\n",grid_search.best_params_


clf = grid_search.best_estimator_

features_selected_bool  =  clf.named_steps['skb'].get_support()
features_selected_list = [x for x, y in zip(features_list[1:],
features_selected_bool ) if y]

print "\nselected features: ", features_selected_list



# ### ADABoost Tuning
# knn_pipeline = Pipeline([("skb", SelectKBest()),
# ("clf", AdaBoostClassifier(DecisionTreeClassifier()))])
#
#
# knn_param_grid = dict(skb__k=range(1, 9),clf__n_estimators =[1,2,50],
# clf__algorithm=["SAMME", "SAMME.R"],
# clf__random_state=[17,1,500])
#
# grid_search = GridSearchCV(knn_pipeline, param_grid=knn_param_grid,
# scoring="recall",cv=5)
#
# grid_search.fit(features_train, labels_train)
# print "\n KNN best estimator: \n", (grid_search.best_estimator_),\
# "\n best score:\n",grid_search.best_score_ ,\
# "\n best params:\n",grid_search.best_params_
#
#
# clf = grid_search.best_estimator_
#
# features_selected_bool  =  clf.named_steps['skb'].get_support()
# features_selected_list = [x for x, y in zip(features_list[1:],
# features_selected_bool ) if y]
#
# print "\nselected features: ", features_selected_list



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.



dump_classifier_and_data(clf, my_dataset, ["poi"]+features_selected_list)
