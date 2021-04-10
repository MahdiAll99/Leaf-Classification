#!/usr/bin/python
#-*- coding: utf-8 -*-
import uuid
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import precision_score as _precision_score
from sklearn.metrics import recall_score as _recall_score
from src.DataProcesser.PreProcesser import PreProcesser
import src.Classifiers as classification


class ResultManager:
    """
    Class that calculates a model's performance statistics. 

    Calculation produced will be used to rank models.
    """
    def __init__(self,cmd = []):
        """
        Parameters
        ==========
        None.
        """
        self._preds = []
        self._truths = []
        self._cmd = cmd
        
        #Needs updating when implementing new metric
        self._methodDict = {
            'Accuracy': self._calculateAccuracy,
            'Precision': self._calculatePrecision,
            'Recall': self._calculateRecall,
            'ConfusionMatrix': self._getConfusionMatrix
        }


    @property
    def preds(self):
        return np.concatenate(self._preds)
    @property
    def truths(self):
        return np.concatenate(self._truths)

    def appendLabels(self,predictions,truths):
        """
        Add new entry to ResultManager

        Parameters
        ==========
        prediction: list -> Predicition labels
        truths: list -> Truth Labels

        Returns
        =======
        void.
        """
        self._preds.append(predictions)
        self._truths.append(truths)

    def _calculateAccuracy(self,):
        """
        Calculate accuracy score from "predictions" and "truth".

        Parameters
        ==========
        None.

        Returns
        =======
        float: Accuracy score in range [0.0, 1.0]
        """
        return '{:.3f}'.format(np.mean(self.preds == self.truths))

    def _calculatePrecision(self):
        """
        Calculate precision score from "predictions" and "truth".

        Parameters
        ==========
        None.

        Returns
        =======
        float: precision score in range [0.0, 1.0]
        """
        return '{:.3f}'.format(_precision_score(self.truths,self.preds,average='macro', zero_division=0.0))
    
    def _calculateRecall(self):
        """
        Calculate recall score from "predictions" and "truth".

        Parameters
        ==========
        None.

        Returns
        =======
        float: recall score in range [0.0, 1.0]
        """
        return '{:.3f}'.format(_recall_score(self.truths,self.preds, average = 'macro', zero_division=0.0))
        


    def _getConfusionMatrix(self, ):
        """
        Calculates confusion matrix from predictions and truth labels.
        
        Parameters
        ==========
        None.

        Returns
        =======
        numpy 2-d array. 
        """
        csv_buffer = StringIO()
        labels = set(self.truths) | set(self.preds)
        cm = _confusion_matrix(self.truths,self.preds,labels = list(labels))
        pd.DataFrame(cm, columns=labels, index=labels).to_csv(csv_buffer)
        return csv_buffer.getvalue()

    def getStatistics(self,):
        """
        Returns all statistics in a JSON object

        Parameters
        ==========
        None.

        Returns
        =======
        float: Accuracy score in range [0.0, 1.0]
        """
        out = {}

        # Get all wanted metric
        for metric in self._cmd:
            out[metric] = self._methodDict[metric]()

        return out
    
    def runTestSet(DataPreProcessingParams:dict, ClassificationParams:dict, StatisticsParams:list, verbose = False):
        """
        Trains on complete training set, predicts on test set and returns predictions with statistics. 

        Parameters
        ==========
        DataPreProcessingParams: dict -> Parameters to send to the DataManagement module.
        ClassificationParams: dict -> Parameters to send to the Classifiers module.
        StatisticsParams: list -> Parameters to send to the Statistician module.

        Returns
        =======
        Results of test set with statistics. All in dictionary format.
        """
        cases = {
            'DataPreProcessingParams':DataPreProcessingParams,
            'ClassificationParams':ClassificationParams,
            'StatisticsParams':StatisticsParams
        }

        # 1. Prepare data
        DP = DataManager(**DataPreProcessingParams)
        DP.importAndPreprocess(label_name = 'species',verbose=verbose)
        DP.split_data(test_ratio=0.1)

        # 2. Create Statistician
        stats = ResultManager(StatisticsParams)

        # 3. Train
        if(verbose): print('Fitting model...',end='')
        clf = classification.getClassifier(**ClassificationParams)
        clf.fit(DP.df_Train,DP.labels_Train)
        if(verbose): print('Done!')

        # 4. Prediction
        if(verbose): print('Prediciting Test Set...',end='')
        predictions = clf.predict(DP.df_Test.values)
        if(verbose): print('Done!')

        # 5. Statistics
        stats.appendLabels(predictions, DP.labels_Test.values)
        return {'predictions':predictions,'truths':DP.labels_Test.values, 'metrics':stats.getStatistics()}