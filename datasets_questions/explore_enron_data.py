#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#REMOVING OUTLIER!!
enron_data.pop("TOTAL", 0)

count = 0
salaryCount =0
mailCount = 0
totalPayments = 0

for key in enron_data.keys():
    #print key
    # print enron_data[key]
    if(enron_data[key]['poi']==True):
        count += 1
        # print enron_data[key]['email_address']

    if(enron_data[key]['salary']!='NaN'):
        salaryCount += 1
    if(enron_data[key]['email_address'].find('@')!=-1):
        mailCount += 1
    if(enron_data[key]['total_payments']=='NaN'):
        totalPayments += 1

print "Total: " ,len(enron_data)
print "No. POI: " ,count
print "Salary count: " ,salaryCount
print "Mail count: " ,mailCount
print "totalPayments NAN: " ,totalPayments


#print "James Prentice: " , enron_data["PRENTICE JAMES"]['total_stock_value']
#print "WESLEY : " , enron_data["COLWELL WESLEY"]
# print "Skilling : " , enron_data["SKILLING JEFFREY K"]
# print "LAY : " , enron_data["LAY KENNETH L"]
# print "FASTOW ANDREW S : " , enron_data["FASTOW ANDREW S"]

def getMissingValueRatio(data):
    """
    data is a pandas dataframe
    """
    ratio = len(data[data=='NaN'])/len(data[0])
    return ratio


feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"

poi  = "poi"
# features_list = [poi, feature_1, feature_2]
features_list = enron_data["FASTOW ANDREW S"].keys()
features_list.remove('email_address')
features_list.remove('other')
features_list.remove('poi')




keys = ['salary', 'to_messages', 'deferral_payments',
'total_payments', 'exercised_stock_options', 'bonus',
'restricted_stock', 'shared_receipt_with_poi',
'restricted_stock_deferred', 'total_stock_value', 'expenses',
'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi',
'poi', 'director_fees', 'deferred_income', 'long_term_incentive',
'email_address', 'from_poi_to_this_person']

# print [poi]+features_list

# data = featureFormat(enron_data, [poi]+features_list )
# poi, finance_features = targetFeatureSplit( data )


# # finance_features.insert(0,[200000.,1000000.])
# print "before scaling: ", stats.describe(finance_features)

edata = getFeaturesAsPandaDict(enron_data,features_list)
# edata.to_csv(path_or_buf="./dataDump.csv")
# print getMissingValueRatio(edata)

# print getFeaturesAsPandaDict(enron_data,features_list).describe()
# print edata['salary']
N = len(edata['salary'])
plt.figure()
# edata['salary'].hist()
fig, ax = plt.subplots()

rects1 = ax.bar(np.arange(N), edata['salary'], 0.35, color='b')

# plt.show()
# def plotFeature(feature, enron_data):
#
#     for key in enron_data.keys():
#         #print key
#         #print enron_data[key]
#         if(enron_data[key]['poi']==True):
#             count += 1


# try:
#     plt.plot(ages, reg.predict(ages), color="blue")
# except NameError:
#     pass
# plt.scatter(ages, net_worths)
# plt.show()
#
# plotFeature("salary",enron_data)











#
