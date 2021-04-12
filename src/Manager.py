#!/usr/bin/python
#-*- coding: utf-8 -*-
import uuid
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import precision_score as _precision_score
from sklearn.metrics import recall_score as _recall_score
from src.DataProcesser.Processer import DataProcesser
import src.Classifiers as classification


class ResultManager:
    """
    Class that calculates a model's performance statistics. 

    Calculation produced will be used to rank models.
    """
    def __init__(self,case = []):
        """
        Parameters
        ==========
        None.
        """
        self._preds = []
        self._truths = []
        self._case = case
        
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
        for metric in self._case:
            out[metric] = self._methodDict[metric]()

        return out
    
def runTestSet(DataPreProcessingParams:dict, ClassificationParams:dict, StatisticsParams:list, verbose = False):
    """
    Trains on complete training set, predicts on test set and returns predictions with statistics. 

    Parameters
    ==========
    DataPreProcessingParams: dict -> Parameters to send to the DataManagement module.
    ClassificationParams: dict -> Parameters to send to the Classifiers module.
    StatisticsParams: list -> Parameters to send to the ResultManager module.

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
    DP = DataProcesser(**DataPreProcessingParams)
    DP.importAndPreprocess(label_name = 'species',verbose=verbose)
    DP.split_data(test_ratio=0.1)

    # 2. Create ResultManager
    RM = ResultManager(StatisticsParams)

    # 3. Train
    if(verbose): print('Fitting model...',end='')
    clf = classification.getClassifier(**ClassificationParams)
    clf.fit(DP.df_Train(),DP.labels_Train())
    if(verbose): print('Done!')

    # 4. Prediction
    if(verbose): 
        print('Prediciting Test Set...',end='')
    predictions = clf.predict(DP.df_Test().values)
    if(verbose): 
        print('Done!')

    # 5. Statistics
    RM.appendLabels(predictions, DP.labels_Test().values)
    return {'predictions':predictions,'truths':DP.labels_Test().values, 'metrics':RM.getStatistics()}

def run(DataPreProcessingParams:dict, ClassificationParams:dict, StatisticsParams:list, verbose = False):
    """
    Launches a machine learning classification evaluation using cross validation

    Parameters
    ==========
    DataPreProcessingParams: dict -> Parameters to send to the DataManagement module.
    ClassificationParams: dict -> Parameters to send to the Classifiers module.
    StatisticsParams: list -> Parameters to send to the ResultManager module.

    Returns
    =======
    Results with hyperparameters. All in dictionary format.
    """
    case = {
        'DataPreProcessingParams':DataPreProcessingParams,
        'ClassificationParams':ClassificationParams,
        'StatisticsParams':StatisticsParams
    }

    # 1. Prepare data
    DP = DataProcesser(**DataPreProcessingParams)
    DP.importAndPreprocess(label_name = 'species',verbose=verbose)

    # 2. Create ResultManager
    RM = ResultManager(StatisticsParams)

    # 3. Perform K-fold
    if(verbose): print('Performing K-fold....',end='')
    for train_data, val_data, train_labels, val_labels in DP.k_fold(k=10):
        
        # 4. Create Classifier
        clf = classification.getClassifier(**ClassificationParams)

        # 5. Fit classifier with training data and labels
        clf.fit(train_data, train_labels)

        # 6. Get predictions
        predictions = clf.predict(val_data)

        # 7. Add labels to ResultManager
        RM.appendLabels(predictions, val_labels.values)

    if(verbose): 
        print('Done!')
    # 8. Calculate average statistics
    statisticsJson = RM.getStatistics()
    
    RM_name = str(uuid.uuid1())
    return {RM_name:{'pipeline':case,'results':statisticsJson}}

if __name__ == '__main__':
    case = {
        "DataPreProcessingParams": {
                "seed": 160743167,
                "cases": [
                    {
                        "method": "StandardScaler",
                        "hyperparams": {}
                    },
                    {
                        "method": "LDA",
                        "hyperparams": {
                            "n_components": 100
                        }
                    }
                ]   
        },
        #CHOOSE ONLY 1.  Note that we index at the end of the list, so no list is actually sent as a param.
        # The reason we did this is to show the different possibilities. 
        'ClassificationParams': [
            {
            'classifier': 'SVM',
            'C': 12,                                    # Regularization parameter.
            'kernel': 'poly',                           # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
            'degree' : 3,                               # Degree of the polynomial kernel function (‘poly’)
            # gamma : {‘scale’, ‘auto’} or float        # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid
            },
            {
            'classifier': 'NeuralNetwork',
            'activation': 'relu',                   # activation {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
            'solver': 'adam',                           # solver {‘lbfgs’, ‘sgd’, ‘adam’}
            'alpha': 0.001,                             # regularization parameter
            'learning_rate': 'invscaling',              # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}
            'max_iter': 1000
            },
            {
            'classifier': 'LogisticRegressionModel',
            'solver': 'liblinear',                      # solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
            'random_state': 0,                          # Control randomness
            'penalty': 'l2',                            # penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
            'tol': 1e-3,                                # Tolerance for stopping criteria
            'C': 2.5,                                   # regularization parameter
            },
            {
            'classifier':'Perceptron',
            'loss': 'perceptron',
            'penalty' : 'l2',
            'alpha': 0.01,                              # Regularization parameter
            'learning_rate': 'invscaling',              # learning_rate {‘constant’,‘optimal’, ‘invscaling’, ‘adaptive’}
            'eta0': 1,                                  # Constant by which the updates are multiplied
            },
            {
            'classifier': 'KernelModel',
            'alpha': 0.0001,
            'kernel': 'rbf',
            'gamma': 0.001                              # gamma defines how much influence a single training example has
            },
            {
            'classifier': 'GenerativeModel'
            }
        ][5], #IMPORTANT! INDEXING HERE, DO NOT ACTUALLY FEED IN A LIST
        'StatisticsParams':[
            'Accuracy','Precision','Recall'#,'ConfusionMatrix'
        ]
    }

    print(run(**case, verbose=True))