#!/usr/bin/python
#-*- coding: utf-8 -*-

from src.Classifiers import Classifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder

class KernelModel(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self._model = KernelRidge(**kwargs)
        self.hyperparams = self._model.get_params()
        self.enc = None

    def fit(self, X, Y):
        """
        Training model with data provided.
        Parameters
        ----------
        X: Pandas DataFrame. #Attribute Values
        Y: Pandas Series.    #Object labels.

        Returns
        ----------
        void.
        """
        X = X.values
        Y = Y.values

        #Verify that the labels can be casted to floats
        try:
            float(Y[0])
        except ValueError:
            self.enc = OneHotEncoder()
            Y = self.enc.fit_transform(Y.reshape(-1,1)).toarray()
        
        self._model.fit(X = X, y = Y)
        
        

    def predict(self, X):
        """
        Returns prediction label for X.
        
        Parameters
        ----------
        X: Pandas DataFrame -> Data to predict value

        Returns
        ----------
        Prediction labels: array like of size (nsamples, [n_features])
        """
        pred = self._model.predict(X)
        if(self.enc is None):
            return pred
        else:
            return self.enc.inverse_transform((pred == pred.max(axis=1, keepdims=1)).astype(float)).reshape(-1)