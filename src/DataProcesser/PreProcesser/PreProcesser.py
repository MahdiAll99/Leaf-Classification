
import json
#This class will manage the data processing
class Preprocesser:
    def __init__(self,**kwargs):
        self._method = None
        self.hyperparams = {}

    def preprocess(self, data):
        
        """ 
        Function that will return the pre-prosessed data
        ----------
        Returns
        ----------
        pre-prosessed data
        """""
        return data

    def jsonify(self, ):
        """ 
        Function that will return the json object
        ----------
        Returns
        ----------
        its json object
        """""

        return {'method':self.__class__.__name__}