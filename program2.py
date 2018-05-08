# -*- coding: utf-8 -*-

import os
import pandas as pd
import sklearn
from sklearn.mixture import GaussianMixture # Gaussian Mixture Models
from sklearn.cluster import KMeans #k-Means Clustering algorithms

from sklearn.datasets import load_files
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# evaluation methods
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import StratifiedKFold

#d = 'C:\\Users\\hkujawska\\Documents\\priv\\UIBMachineLearning\\AS2\\*'
#os.listdir(d)
#os.getcwd()
#wheatData = load_files("C:\\Users\\hkujawska\\Documents\\priv\\UIBMachineLearning\\AS2\\seeds_dataset.txt")
#print('yes!')
from sklearn import datasets



in_file = "seeds_dataset.csv"
def load_dataset():
    in_file = "seeds_dataset.txt"
    wheatData = datasets.load_files(container_path="C:\\Users\\hkujawska\\Documents\\priv\\UIBMachineLearning\\AS2",                    categories=None, load_content=True,
                       encoding='utf-8', shuffle=True, random_state=42)
#load_iris()
    print('type',type(wheatData))
#    colnames = [ 'area A', 'perimeter P', 'compactness C = 4*pi*A/P^2','length of kernel','width of kernel','asymmetry coefficient', 'length of kernel groove', 'class']
#    wheatData = pd.read_csv(in_file,sep='\t', names = colnames);
#    # 1. print dimensions
#   [col,row] = wheatData.shape
#    print('Dataset has {0} columns and {1} rows'.format(col, row))
#    # 2. print top 5 lines
#    print(wheatData.head(5))

    ds = wheatData.values
    data = ds[:,0:5]
    print('data',data)
    
    print('wheatData.values',wheatData.values,)
    
    #create list of algorithms
    algorithmName = ['GMM','kMeansClustering']
    algorithm = [GaussianMixture(), KMeans()]
    
    PairsList= {}
    for i in range(len(algorithmName)):
        PairsList[algorithmName[i]] = algorithm[i]
    
    l=[]
    for i in range(len(algorithmName)):
        l.append((algorithmName[i], algorithm[i]))
        
    
     #  the .values function to extract all the values of the DataFrame and then split it into features X and target classes Y.
#    df = wheatData.values
#    featuresX =df[:, 0:7]
#    featuresY = df[:, -1]
    
    # Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
    skf = StratifiedKFold(n_splits=4)
    
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(wheatData.data  , wheatData.target)))
    print('train_index, test_index',train_index, test_index)
    
    #split the dataset into training and validation sets using train test split(): half for training, and the other half for validation
#    XtrainSet, XtestSet, YtrainSet, YtestSet = train_test_split(featuresX , featuresY, test_size=0.2)
    
#    print('XtrainSet',XtrainSet,'XtestSet', XtestSet, 'YtrainSet', YtrainSet,'YtestSet', YtestSet)
    

    # number of components (elements)
    elementsRange =  7;
   
    wheat=[]
    #types of Gaussian Mixture Models 
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for element in range(1,elementsRange-1):
             # Fit a Gaussian mixture with EM
             gmm = GaussianMixture(n_components=elementsRange,
                                      covariance_type=cv_type)
             gmm.fit(XtrainSet)
             wheat.append(gmm.wheat(X))
             
             # make prediction using the testing sets
             Ypred = gmm.predict(XtestSet)
             
             #evaluate the performance of this model on the validation dataset by printing out the result of running                classification_report()
             evaluation = classification_report(YtestSet, Ypred)
             print('Evaluation:',evaluation)
            
            
            
    return (wheatData)

load_dataset()

'''
 apply both the Gaussian Mixture Models and the k-Means Clustering algorithms on this dataset
'''