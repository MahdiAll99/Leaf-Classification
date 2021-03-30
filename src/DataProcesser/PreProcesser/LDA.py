
import pandas as pd
from src.DataManagement.Preprocesser import Preprocesser
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis as LDA

class LDA(Preprocesser):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = LDA(**kwargs)
        self.hyperparams = self._method.get_params
        
    def preprocess(self, data):
        """ 
        Function that will Fit to data, then transform it using Linear Discriminant Analysis
        ----------
        Returns
        ----------
        Data frame with transformed data
        """"
        data_transformed = self._method.fit_transform(data)
        return pd.DataFrame(data_transformed,columns=['LDA_axis%d'%i for i in range(len(self._method.singular_values_))], index = data.index)

    def jsonify(self):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out