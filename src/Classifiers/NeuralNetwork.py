#!/usr/bin/python
#-*- coding: utf-8 -*-

from sklearn.neural_network import MLPClassifier
from src.Classifiers import Classifier

class NeuralNetwork(Classifier):
    def __init__(self, **kwargs):
        super().__init__()
        self._model = MLPClassifier(**kwargs)

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

        self._model.fit(X=X, y=Y)

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

        return pred


