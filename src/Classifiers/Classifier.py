#!/usr/bin/python
#-*- coding: utf-8 -*-
import pandas as pd
import sys

def getClassifier(classifier:str, **hyperparams ):
    """ 
    Searches environement for definition of classifier

    Parameters
    ----------
    classifier:str -> Name of the classifier class.
    hyperparams: dict -> Arguments to classifier

    Returns
    ----------
    Instance of chosen classifier
    """
    return getattr(sys.modules[__package__],classifier)(**hyperparams)

class Classifier:
    """Base Class for all classifiers."""
    def __init__(self):
        self._model = None

    def fit(self, data:pd.DataFrame):
        """Training model"""
        pass

    def predict(self, X:pd.DataFrame):
        """Returns prediction label for X."""
        pass


if __name__ == '__main__':
    from src.Classifiers import *

    #TODO : Create and process data
    train_data = []
    train_labels = []

    test_data = []
    test_labels =[]

    #Create a classifier
    clf = getClassifier(classifier='KernelMethod', alpha = 0.001, kernel = 'rbf')

    
    #Fit and predict
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)

    #Calculate accuracy score
    #In project, this will be handled by the Statistician package
    accuracyList = (predictions == test_labels)
    accuracy = sum(accuracyList) / len(accuracyList)
    print('Accuracy: {:.3f}%'.format(accuracy * 100))
