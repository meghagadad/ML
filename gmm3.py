# -*- coding: utf-8 -*-
"""
Created on Tue May  8 18:23:54 2018

@author: hkujawska
"""

#wheatData = datasets.load_files(container_path="C:\\Users\\hkujawska\\Documents\\priv\\UIBMachineLearning\\AS2",                        categories=None, load_content=True, encoding='utf-8', shuffle=True, random_state=42)
#
## Break up the dataset into non-overlapping training (75%) and testing
## (25%) sets.
#skf = StratifiedKFold(n_splits=4)
#print('skf',skf)
## Only take the first fold.
#print('wheatData.data',wheatData.target)
#l = wheatData.data
import numpy as np

with open ('seeds_dataset.csv', 'r') as f:
    data1= f.read().split('\n')
    data1 = np.array(data)

print('data',type(data))
target= [1,2,3,]

train_index, test_index = next(iter(skf.split(data1,target )))

#train_index, test_index = next(iter(skf.split(wheatData.data, wheatData.target),'\t'))
#
