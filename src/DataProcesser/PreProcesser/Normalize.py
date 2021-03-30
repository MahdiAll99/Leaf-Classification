
import pandas as pd
from src.DataManagement.Preprocesser import Preprocesser
from sklearn.preprocessing import MinMaxScaler

class Normalize(Preprocesser):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = MinMaxScaler(**kwargs)
        self.hyperparams = self._method.get_params()
        
    def preprocess(self, data):
        """ 
        Function that will Fit to data, then transform it using the MinMax transformation
        ----------
        Returns
        ----------
        Data frame with transformed data
        """"
        return pd.DataFrame(self._method.fit_transform(data), columns = data.columns, index = data.index)

    def jsonify(self):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out