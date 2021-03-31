
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
        """ 
        Function that will prepare Image DataFrame and then insert it into existing DataFrame
        ----------
        Returns
        ----------
        Data frame with image included
        """"
        DATA_PATH = os.path.dirname(os.path.realpath(__file__))
        DATA_PATH = DATA_PATH.parent.absolute() / 'data'
        IMAGES_PATH = DATA_PATH + self.hyperparams['Images_Path']

        columns = ['Pixel_%d'%i for i in range(np.prod((64,64)))]
        img_df = pd.DataFrame(columns = columns)
        
        for filename in os.listdir(IMAGES_PATH):
            img = Image.open(IMAGES_PATH + filename)
            img = np.array(img.resize((64,64))).flatten()
            img_df.loc[int(filename.strip('.jpg'))] = img

    def jsonify(self, ):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out