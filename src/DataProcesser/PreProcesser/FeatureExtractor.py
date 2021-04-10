
from src.DataProcesser.PreProcesser import Preprocesser
import re

#This class inherits from Preprocesser

class FeatureExtraction(Preprocesser):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = None
        self.hyperparams = {}
        self.hyperparams.update(**kwargs)

    def preprocess(self, data):
        #Check to see if all columns and indexes are requested.
        out = data
        if ('columns' in self.hyperparams):
            r = re.compile(self.hyperparams['columns'])
            out = out[list(filter(r.match,out.columns))]
        if ('index' in self.hyperparams):
            r = re.compile(self.hyperparams['index'])
            out = out[list(filter(r.match,out.index))]
        return out

    def jsonify(self, ):
        #return the pre-processsed data as a json
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out