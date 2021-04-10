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

    def importData(self, label_name, filepath = RAWDATA_PATH):
        """Import data from file"""
        #Obtain features from train.csv
        # 1. Obtain csv of features
        self._df = pd.read_csv(filepath, index_col = 'id')
        # 2. Seperate labels
        self.label_name = label_name
        self._labels = self._df[self.label_name]
        # Clean Up RAM
        del self._df[label_name]
    
    def preprocess(self,verbose=False):
        """Preprocess data. Save Preprocessed data to file."""
        # Convert cmd into a list of Preprocessing Strategies
        if(verbose): 
            print('Pre-Processing.....')
        for i,cmd in enumerate(self.cmds):
            if(verbose): 
                #Method used in data processing
                print('\tMethod #%d:'%i,cmd)

            strategy = getattr(preproc,cmd['method'])(**cmd['hyperparams'])
            self._df = strategy.preprocess(self._df)
        if(verbose):  
            print('Done!')

    def saveData(self, savepath = PROCESSEDDATA_FOLDER, verbose = False):
        # 1.Load already saved file
        if(not os.path.isfile(savepath + PROCESSED_FILENAME)):
            preprocessedJS = {}
        else:
            with open(savepath + PROCESSED_FILENAME) as f:
                preprocessedJS = json.load(f)
        
        # 2.Add commands with uuid that referenced saved data
        uuidfilename = str(uuid.uuid1()) + '.csv'
        preprocessedJS[uuidfilename] = self.cmds
        
        # 3.save :
        if(verbose): 
            print('Saving preprocessed data to %s'%(savepath + uuidfilename))
        with open(savepath + PROCESSED_FILENAME,'w') as f:
            json.dump(preprocessedJS, f,indent=4)
        df_tobesaved = self._df.copy()
        df_tobesaved[self.label_name] = self._labels
        df_tobesaved.to_csv(savepath + uuidfilename)

    #This method is quite the implementation of the two previous methods (proprocess and importData)
    def importAndPreprocess(self,label_name,filepath = RAWDATA_PATH, savepath = PROCESSEDDATA_FOLDER, verbose=False):
        """ 
        Checks if pre-processed data already saved. 
        Loads and returns pre-processed data if so. 
        Else, generates preprocessed data and saves it. 
        """
        # 1. Check to see if data has already been preprocessed
        if(os.path.isfile(savepath + PROCESSED_FILENAME)):
            with open(savepath + PROCESSED_FILENAME) as f:
                preprocessedJS = json.load(f)
            for key in preprocessedJS:
                if(preprocessedJS[key] == self.cmds):
                    preprocessedPath = savepath + key
                    if(verbose): 
                        print('Loading saved preprocessed data from %s'%(preprocessedPath))
                    return self.importData(label_name = label_name,filepath = preprocessedPath)
        # 2.Else, load raw data, preprocess and save
        self.importData(label_name=label_name,filepath=filepath)
        self.preprocess(verbose=verbose)
        self.saveData(savepath=savepath, verbose=verbose)
    
    def split_data(self,test_ratio = 0.1):
        #Seperates dataset into train sets and test sets (10%)
        self._train_indexes, self._test_indexes = train_test_split(self._df.index, random_state = self._seed)