
from src.DataManagement.Preprocesser import Preprocesser
import pathlib
import os
import pandas as pd
from PIL import Image
import numpy as np

#This script is to import images and insert it in our DataFrame

class ImageToDF(Preprocesser):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = None
        self.hyperparams = None
        self.hyperparams.update(**kwargs)

    def preprocess(self, data):
        pass

    def jsonify(self, ):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out