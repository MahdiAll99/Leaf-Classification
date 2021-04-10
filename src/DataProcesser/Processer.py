#!/usr/bin/python
#-*- coding: utf-8 -*-
import pandas as pd
import os
import pathlib
from src.DataProcesser import Preprocessing as preproc
import json
import uuid
import numpy as np
from sklearn.model_selection import KFold as _KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/../../data/'
RAWDATA_PATH = DATA_PATH + '/raw/train.csv'
PROCESSEDDATA_FOLDER = DATA_PATH + '/processed/'
PROCESSED_FILENAME = 'processed.json'

class DataProcesser:
    """
    _df : data frame
    label_name : name of the label(column) that we will extract from the csv file
    _labels : labels extracted from csv dile
    _seed : used to initialize the random number generator
    cmds : contain the method that will be used and hyperparameters
    """
    def __init__(self, seed = 0, cmds = []):
        self._df = None
        self._labels = None
        self.label_name = ""
        self._seed = seed
        self.cmds = cmds

        self._train_indexes = None 
        self._test_indexes = None
    
    @property
    def train_indexes(self):
        if(self._train_indexes is None): return self._df.index
        else:                            return self._train_indexes
    @train_indexes.setter
    def train_indexes_setter(self,val):
        self._train_indexes = val
    @property
    def test_indexes(self):
        if(self._test_indexes is None):  return []
        else:                            return self._test_indexes
    @test_indexes.setter
    def train_indexes_setter(self,val):
        self._test_indexes = val

    def df(self):
        return self._df
    def df_Train(self):
        return self._df.loc[self.train_indexes]
    def df_Test(self):
        return self._df.loc[self.test_indexes]
    def labels(self):
        return self._labels
    def labels_Train(self):
        return self._labels.loc[self.train_indexes]
    def labels_Test(self):
        return self._labels.loc[self.test_indexes]