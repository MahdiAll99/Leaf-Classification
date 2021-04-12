"""
MAHDI AIT LHAJ LOUTFI (aitm2302)
YOVAN TUROCTTE (tury1903)
"""
from src.Manager import run
import numpy as np
import itertools
import random
import functools
import pathlib
import os
import json

SAVEPATH = str(pathlib.Path(__file__).parent.absolute()) + '/results/' #Path to save JSON resutls
RESULTS_FILENAME =  'results.json'

# Define a classifier gridsearch generator
class ClassifierGridSearch():
    def __init__(self,classifier,**hyperparams):
        self.classifier = classifier
        self.hyperparams = hyperparams

    def gridsearchGenerator(self):
        """Perform girdSearch of hyperparams"""
        keys = self.hyperparams.keys()
        vals = self.hyperparams.values()
        for validTuple in itertools.product(*vals):
            yield {'classifier':self.classifier, **dict(zip(keys,validTuple))}

    def __len__(self):
        vals = self.hyperparams.values()
        return functools.reduce(lambda count,x: count*len(x),vals, 1)

# Define a DataManager gridsearch generator
class DataManagerGridSearch():
    def __init__(self,seed,featureAugmenters,scalers,featureExtracters):
        self.seed = seed
        self.scalers = scalers
        self.featureAugmenters = featureAugmenters
        self.featureExtracters = featureExtracters

    def gridsearch(self):
        """Returns a list of hyperparams for each runned case"""
        cases = []
        if(len(self.featureAugmenters) > 0): cases.append(self.featureAugmenters)
        if(len(self.scalers) > 0): cases.append(self.scalers)
        if(len(self.featureExtracters) > 0): cases.append(self.featureExtracters)
        return list(map(lambda x: {'seed':self.seed, 'cases':list(x)}, list(itertools.product(*cases))))

def saveDict(obj,savepath = SAVEPATH + RESULTS_FILENAME):
    with open(savepath, 'w') as f:
        json.dump(obj,f,indent=4)


if __name__ == '__main__':
    # 0. Load results json file, if it exists.
    if(os.path.isfile(SAVEPATH + RESULTS_FILENAME)):
            with open(SAVEPATH + RESULTS_FILENAME) as f:
                resultsJson = json.load(f)
    else:
        resultsJson = {}

    # 2. Create objet to perform grid serach over preprocessing methods. 
    cases = DataManagerGridSearch(seed = 16082604, 
                                featureAugmenters = [], #No data augmentation because it takes too much place on disk
                                scalers = [
                                    {   'method':'Normalize',
                                        'hyperparams':{}
                                    },
                                    {   'method':'StandardScaler',
                                        'hyperparams':{}
                                    }
                                ],
                                featureExtracters = [
                                    {   'method':'LDA',
                                        'hyperparams':{'n_components':100}
                                    },
                                    {   'method':'FeatureExtraction',
                                        'hyperparams':{'columns':r'^\w*\d*[02468]$'} #Only the even feature.
                                    }
                                ]
                                ) 

    #3. Must create a gridsearch generator for every classifier
    cgsKernelModel = ClassifierGridSearch( classifier='KernelModel', 
                                            alpha= np.logspace(-9, np.log10(2), 20), 
                                            kernel = ['rbf','linear','poly'], 
                                            gamma =  np.logspace(-9,np.log10(2), 20)
                                            )
    cgsGenerativeModel = ClassifierGridSearch( classifier='GenerativeModel')
    cgsLogisticRegression = ClassifierGridSearch(   classifier = 'LogisticRegressionModel',
                                                    solver= ['liblinear'],                       # solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
                                                    random_state = [0],                          # Control randomness
                                                    penalty = ['l2'],                            # penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
                                                    tol = np.logspace(-4,np.log10(2), num=20),   # Tolerance for stopping criteria
                                                    C = np.logspace(-4,4,num=20),                # regularization parameter
                                                )
    cgsNeuralNetwork = ClassifierGridSearch(    classifier = 'NeuralNetwork',
                                                hidden_layer_sizes=[(100), (200), (300)],
                                                activation = ['relu','tanh','logistic'],     # activation {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
                                                solver = ['adam'],                           # solver {‘lbfgs’, ‘sgd’, ‘adam’}
                                                alpha = np.logspace(-9,np.log10(2),num=20),  # regularization parameter
                                                learning_rate = ['invscaling'],              # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}
                                                max_iter= [1000]
                                                )
    cgsPerceptron = ClassifierGridSearch(   classifier = 'Perceptron',
                                            loss = ['perceptron'],
                                            penalty = ['l2'],
                                            alpha =  np.logspace(-9,np.log10(2),num=20), # Regularization parameter
                                            learning_rate = ['invscaling'],              # learning_rate {‘constant’,‘optimal’, ‘invscaling’, ‘adaptive’}
                                            eta0 = [1],                                  # Constant by which the updates are multiplied
                                                )
    cgsSVM = ClassifierGridSearch(  classifier = 'SVM',
                                    C = np.logspace(-4,4,num=20),               # Regularization parameter.
                                    kernel = ['rbf','linear','poly','sigmoid'], # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
                                    degree = [2],                               # Degree of the polynomial kernel function (‘poly’)
                                    gamma =  np.logspace(-9,np.log10(2), 20),   # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid
                                                )

    # 4. Create Generator that generates hyperparameter values
    cgs = itertools.chain(*map(lambda x: x.gridsearchGenerator(), [cgsKernelModel, cgsGenerativeModel, cgsLogisticRegression, cgsNeuralNetwork, cgsPerceptron, cgsSVM]))
    gen = itertools.product(cgs,cases.gridsearch())
    
    # 5. Advance generator to right place. Necessary if mid grid-search crash so as not to start all over again.
    for _ in range(len(resultsJson)):
        next(gen)

    # 6. For each hyperparam combination that hasnt been run, run model and update results
    hasChanged = False
    for i,(c,d) in enumerate(gen, start = len(resultsJson)):
        #Checkpoint
        print('Combinaison Num-%d'%i)
        saveDict(resultsJson)

        #7. Run and update results!!
        resultsJson.update(run(DataPreProcessingParams = d, ClassificationParams = c, StatisticsParams= ['Accuracy','Precision','Recall'], verbose = False))
        hasChanged = True

    #Final save of values
    if(hasChanged):
        saveDict(resultsJson)

    #Results analysis is done in ./notebook/ResultsAnalysis.ipynb


