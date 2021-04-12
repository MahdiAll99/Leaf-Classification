
import pandas as pd
from src.DataProcesser.PreProcesser import Preprocesser
from sklearn.preprocessing import PolynomialFeatures as PF

class PolynomialFeatures(Preprocesser):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = PF(**kwargs)
        self.hyperparams = self._method.get_params()
        
    def preprocess(self, data):
        """ 
        Function that will Fit to data, then Generate polynomial and interaction features it using PolynomialFeatures
        ----------
        Returns
        ----------
        Data frame with transformed/Generated data
        """
        return pd.DataFrame(self._method.fit_transform(data), columns = self._method.get_feature_names(data.columns), index = data.index)

    def jsonify(self):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out