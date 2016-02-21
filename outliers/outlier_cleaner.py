#!/usr/bin/python
from sklearn import linear_model
from math import *

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    ### your code goes here

    reg = linear_model.LinearRegression()


    reg.fit(ages,net_worths)

    diffs = list()
    # print predictions[0][0]
    for i in range(0,len(predictions)):
        # print fabs(predictions[i][0]-net_worths[i][0])
        diffs.append((ages[i][0],net_worths[i][0],predictions[i][0]-net_worths[i][0]))

    # print diffs[0]
    diffs = sorted(diffs,key=lambda tup: fabs(tup[2]))
    cleaned_data = diffs[0:81]
    # print len(cleaned_data)

    return cleaned_data










###
