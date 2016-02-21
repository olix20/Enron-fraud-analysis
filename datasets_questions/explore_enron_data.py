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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

count = 0
salaryCount =0
mailCount = 0
totalPayments = 0

for key in enron_data.keys():
    #print key
    #print enron_data[key]
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

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def plotFeature(feature, enron_data):

    for key in enron_data.keys():
        #print key
        #print enron_data[key]
        if(enron_data[key]['poi']==True):
            count += 1


# try:
#     plt.plot(ages, reg.predict(ages), color="blue")
# except NameError:
#     pass
# plt.scatter(ages, net_worths)
# plt.show()
#
plotFeature("salary",enron_data)











#
