#!/usr/bin/python
#-*- coding: utf-8 -*-
import uuid
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import precision_score as _precision_score
from sklearn.metrics import recall_score as _recall_score
from src.DataManagement.Manager import DataManager
import src.Classifiers as classification


def _calculateAccuracy(preds,truths):
        """
        Calculate accuracy score from "predictions" and "truth".

        Parameters
        ==========
        Preds : Predictions
        Truths : True Results.

        Returns
        =======
        float: Accuracy score in range [0.0, 1.0]
        """
        return '{:.3f}'.format(np.mean(preds == truths))

    def _calculatePrecision(preds, truths, average=None, zero_division=None):
        """
        Calculate precision score from "predictions" and "truth".

        Parameters
        ==========
        Preds : Predictions
        Truths : True Results.
        average : {'micro', 'macro', 'samples', 'weighted', 'binary'} default='binary'
                   This parameter is required for multiclass/multilabel targets.
        zero_division : 0 or 1 .
                        Sets the value to return when there is a zero division.

        Returns
        =======
        float: precision score in range [0.0, 1.0]
        """
        return '{:.3f}'.format(_precision_score(truths,preds,average='macro', zero_division=0.0))
    
    def _calculateRecall(truths, preds, average=None, zero_division=None):
        """
        Calculate recall score from "predictions" and "truth".

        Parameters
        ==========
        Preds : Predictions
        Truths : True Results.

        Returns
        =======
        float: recall score in range [0.0, 1.0]
        """
        return '{:.3f}'.format(_recall_score(self.truths,self.preds, average = 'macro', zero_division=0.0))
        


    def _getConfusionMatrix(preds, truths):
        """
        Calculates confusion matrix from predictions and truth labels.
        
        Parameters
        ==========
        Preds : Predictions
        Truths : True Results

        Returns
        =======
        numpy 2-d array. 
        """
        csv_buffer = StringIO()
        labels = set(self.truths) | set(self.preds)
        cm = _confusion_matrix(self.truths,self.preds,labels = list(labels))
        pd.DataFrame(cm, columns=labels, index=labels).to_csv(csv_buffer)
        return csv_buffer.getvalue()

    def getResults(cases, methodDict):
        """
        Returns all results in a JSON object

        Parameters
        ==========
        cases : all test case used
        methodDict : contains informations about Accuracy, Precision, Recall, ConfusionMatrix 

        Returns
        =======
        float: Accuracy score in range [0.0, 1.0]
        """
        out = {}

        # Get all wanted metric
        for metric in cases:
            out[metric] = methodDict[metric]()

        return out