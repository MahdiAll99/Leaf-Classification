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
        None.

        Returns
        =======
        float: Accuracy score in range [0.0, 1.0]
        """
        return '{:.3f}'.format(np.mean(preds == truths))

    def _calculatePrecision(preds,truths,average=None,zero_division=None):
        """
        Calculate precision score from "predictions" and "truth".

        Parameters
        ==========
        None.

        Returns
        =======
        float: precision score in range [0.0, 1.0]
        """
        return '{:.3f}'.format(_precision_score(truths,preds,average='macro', zero_division=0.0))