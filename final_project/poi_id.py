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

# features_list = ['poi','salary', "total_stock_value","total_payments",
# "shared_receipt_with_poi","restricted_stock","expenses",
# "exercised_stock_options","total_correspondence_with_poi",
# "bonus"]
# print list(xrange(10))

features_list = ['poi','salary', 'to_messages', 'deferral_payments',
'total_payments', 'exercised_stock_options', 'bonus',
'restricted_stock', 'shared_receipt_with_poi',
'restricted_stock_deferred', 'total_stock_value', 'expenses',
'loan_advances', 'from_messages',  'from_this_person_to_poi',
'director_fees', 'deferred_income', 'long_term_incentive',
 'from_poi_to_this_person']

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

# Scale ..

# selection =  SelectKBest(k=7)
# # selection.fit(features,labels)
# # print selection.scores_
# # print "SelectKBest values: \n" ,selection.get_support()
# # print
# #
# features_selected_list = [x for x, y in zip(features_list[1:], selection.get_support()) if y]
# print features_selected_list
# # The 5th feature is not comparable to the first 4 so we'll drop
#
# pca = PCA(n_components=5)
# # pca.fit(features)
# # print "PCA values: \n",(pca.explained_variance_ratio_)
# There is only 1 significant vector (0.81)

# Build estimator from PCA and Univariate selection:
# combined_features = FeatureUnion([("pca", PCA(n_components=1)),
# ("univ_select", SelectKBest(k=4))])

# Use combined features to transform dataset:
# X_features = combined_features.fit(features, labels).transform(features)
#


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=40))


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

knn_clf = KNeighborsClassifier()




### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

# final_features = ['poi','salary', "total_stock_value","restricted_stock",
# "exercised_stock_options","bonus","total_correspondence_with_poi"]
# [ True  True False False  True False  True False  True]

# final_features = ["poi"]+features_selected_list
# data = featureFormat(my_dataset,features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)


# scaler = MinMaxScaler()
# features =  scaler.fit_transform(features)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)



scores = ['precision', 'recall']

## KNN Tuning
pipeline = Pipeline([("features", SelectKBest()),("scaler", MinMaxScaler()), ("clf", knn_clf)])
# # # features__pca__n_components=[1], features__univ_select__k=[1, 2,3,4],svm__C=[10],clf__criterion=['gini', 'entropy'],
#
#
knn_param_grid = dict(features__k=range(1, 15),clf__n_neighbors=[1,5,10,15,20],
clf__weights=['uniform','distance'], clf__algorithm=['ball_tree','kd_tree'],
clf__leaf_size=[1,5,10])
#
grid_search = GridSearchCV(pipeline, param_grid=knn_param_grid)
grid_search.fit(features_train, labels_train)
print "\n KNN best estimator: \n", (grid_search.best_estimator_), "\n best score:\n",grid_search.best_score_ ,"\n best params:\n",grid_search.best_params_


clf = grid_search.best_estimator_
# clf.fit(features_train,labels_train )
print clf.score(features_test,labels_test)


selection =  SelectKBest(k=grid_search.best_params_['features__k'])
selection.fit(features_train,labels_train)
features_selected_list = [x for x, y in zip(features_list[1:], selection.get_support()) if y]

clf = KNeighborsClassifier()
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, ["poi"]+features_selected_list)
