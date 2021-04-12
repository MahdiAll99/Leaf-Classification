#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import pandas as pd
import os
import pathlib
from src.DataProcesser import PreProcesser as preproc
import json
import uuid
import numpy as np
from sklearn.model_selection import KFold as _KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



DATA_PATH = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/data/'
RAWDATA_PATH = DATA_PATH + '/raw/train.csv'
PROCESSEDDATA_FOLDER = DATA_PATH + '/processed/'
PROCESSED_FILENAME = 'processed.json'

class DataProcesser:
    """
    _df : data frame
    label_name : name of the label(column) that we will extract from the csv file
    _labels : labels extracted from csv dile
    _seed : used to initialize the random number generator
    cases : contain the method that will be used and hyperparameters
    """
    def __init__(self, seed = 0, cases = []):
        self._df = None
        self._labels = None
        self.label_name = ""
        self._seed = seed
        self.cases = cases

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
        for i,cmd in enumerate(self.cases):
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
        preprocessedJS[uuidfilename] = self.cases
        
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
                if(preprocessedJS[key] == self.cases):
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

    def k_fold(self, k):
        """Returns generator that produces training sets and validation sets"""

        # 1.Splits data
        if(self._train_indexes is None):    
            self.split_data(test_ratio=0.1)

        # 2.Creates KFold generator
        kf = _KFold(n_splits=k,shuffle=True,random_state=self._seed)
        
        # 3.Create generator that splits the existing data and labels for train and validation sets
        for train_idxes, val_idxes in kf.split(self.df_Train().to_numpy(), self.labels_Train().to_numpy()):
            train_data = self.df_Train().iloc[train_idxes]
            val_data = self.df_Train().iloc[val_idxes]
            train_labels = self.labels_Train.iloc[train_idxes]
            val_labels = self.labels_Train.iloc[val_idxes]
            yield  train_data, val_data, train_labels, val_labels


    def CreateCase(self,method = "",**kwargs):
        """ Function that converts arguments to JSON that will be added as parameter to calculation. """
        hyperparams = {**kwargs}
        out = {'method' : method, 'hyperparams':hyperparams}
        self.cases.append(out)

    def setSeed(self, seed):
        """Set the seed attributre for the random generator."""
        self._seed = seed

if __name__ == '__main__':
    METHOD = 0 
    if(METHOD == 0):
        #Testing PolynomialFeatures, Normalize, PCA
        DP = DataProcesser()
        #Create some test cases :
        DP.CreateCase(method='Normalize')
        DP.CreateCase(method='PolynomialFeatures',degree = 2)
        DP.CreateCase(method='LDA',n_components=None)

        #Testing Importing and Preprocessing Data
        DP.importAndPreprocess(label_name = 'species')

        #Spliting data into Train & Validation & Test sets
        DP.setSeed(0) #Important in order to always have same test set
        DP.split_data(test_ratio=0.1)
        print("Number of Test observations:", len(DP.df_Test))

        #Getting K-fold datasets
        for i, (X_train, X_val, Y_train, Y_val) in enumerate(DP.k_fold(k=10)):
            print('K =', i)
            print('\tNumber of Train observations:', len(X_train))
            print('\tNumber of Validation observations:', len(X_val))