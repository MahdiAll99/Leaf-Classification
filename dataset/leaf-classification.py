"""
MAHDI AIT LHAJ LOUTFI (aitm2302)
YOVAN TUROCTTE (tury1903)
"""
import zipfile
import pathlib
import os

SCRIPT_PATH = str(pathlib.Path(__file__).parent.absolute())
DATASET_PATH = SCRIPT_PATH + '/leaf-classification.zip'
OUT_DIRECTORY = SCRIPT_PATH + '/../data/raw/' #data folder should/will be in root folder!

# If OUT_DIRECTORY folder has a data folder in it, then the dataset has already been extracted
if(not os.path.exists(SCRIPT_PATH + '/../data')):
    os.mkdir(SCRIPT_PATH + '/../data')
    os.mkdir(SCRIPT_PATH + '/../data/raw/')
    os.mkdir(SCRIPT_PATH + '/../data/processed/')
else:
    if(not os.path.exists(SCRIPT_PATH + '/../data/processed/')):
        os.mkdir(SCRIPT_PATH + '/../data/processed/')
    if(not os.path.exists(SCRIPT_PATH + '/../data/raw/')):
        os.mkdir(SCRIPT_PATH + '/../data/raw/')
    #Check to see if dataset has already been extracted
    else:
        filenames = [fn for fn in os.listdir(SCRIPT_PATH + '/../data/raw/') if '.git' not in fn]
        if( len(filenames) > 0):
            print("Leaf-classification dataset has already been extracted.")
            quit()

        else :
            #Extract leaf-classification dataset into ../data/raw/
            with zipfile.ZipFile(DATASET_PATH) as zf:
                zf.extractall(OUT_DIRECTORY)

            #Extract all zips within the zip file
            for filename in os.listdir(OUT_DIRECTORY):
                if(filename.endswith('.zip')):
                    #Extract zipfile
                    with zipfile.ZipFile(OUT_DIRECTORY + filename) as zf:
                        zf.extractall(OUT_DIRECTORY)
                    
                    #Delete zipfile as the contents have been extracted
                    os.remove(OUT_DIRECTORY + filename)

            print('Leaf-classification dataset has been extracted.')