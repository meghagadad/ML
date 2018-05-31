# -*- coding: utf-8 -*-
"""
Created on Sat May 26 11:35:45 2018

@author: hkujawska
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:18:09 2018
@author: hkujawska
"""

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy.ma as ma

from sklearn import mixture
from sklearn.model_selection import train_test_split
# evaluation methods
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

#ploting ellipses
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 2, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[i][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[i][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[i]
        v, w = np.linalg.eigh(covariances)

        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(min(X[:, 0])-0.5, max(X[:, 0])+0.5)
    plt.ylim(min(X[:, 1])-0.5, max(X[:, 1])+0.5)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


color_iter = itertools.cycle(['navy', 'turquoise', 'darkorange', 'orange'])
colors = ['navy', 'turquoise', 'darkorange', 'orange'
          ]
# read file
in_file = 'seeds_dataset.txt'
colnames = ['area A', 'perimeter P', 'compactness C = 4*pi*A/P^2', 'length of kernel', 'width of kernel',
            'asymmetry coefficient', 'length of kernel groove', 'class']
wheatData = pd.read_csv(in_file, delim_whitespace=True, names=colnames);

# take two features and target
dF = wheatData.values
featureA = 0
featureB = 3
target = 7
data = dF[:, [featureA, featureB]]
Y = dF[:, target]
[row, col] = data.shape
#keep randomcy
np.random.seed(0)
SampleList = []                 # List of randomly chossed rows
SampleArray = np.array([])      # Array of randomly choosed rows
for i in range(2):
    randomRow = np.random.choice(row, 1)
    C = np.array(dF[int(randomRow), [featureA, featureB]])
    SampleArray = np.append(SampleArray, C)
    np.array(SampleList.append(C))

# X is dataframe with features
X = (np.array(dF[:, [featureA, featureB]]))
X = ma.masked_invalid(X)
Y = ma.masked_invalid(Y)

n_samples = 210
n_classes = 3
cv_types = ['full','spherical','diag','tied' ]


for n, (cv_type, color) in enumerate(zip(cv_types, colors)):
    gmm = mixture.GaussianMixture(n_components=n_classes, covariance_type=cv_type).fit(X)
  
    ##split the dataset into training and validation sets using train test split()
    XtrainSet, XtestSet, YtrainSet, YtestSet = train_test_split(X, Y,test_size=0.25)
    
    #train model
    gmm.fit(XtrainSet,YtrainSet)
    
    ## make prediction using the testing sets
    Ypred = gmm.predict(XtestSet).reshape(-1, 1)
    
    ##evaluate the performance of this model on the validation dataset by printing out the result of running classification_report()
    evaluation = classification_report(YtestSet.round(), Ypred)
    print('Evaluation:' + str(cv_type),evaluation)
    accuracy = accuracy_score(YtestSet.round(), Ypred, normalize=False)
#    print('YtestSet', YtestSet, Ypred)
#    r2_score = r2_score(YtestSet, Ypred)
    print('The accuracy is:{0}. '.format(round(accuracy,2)))
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, n,
             'Gaussian Mixture ' + str(cv_type))
    for n, color in enumerate(colors):    
        data = X[Y == n+1 ]
        plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)
 
    plt.show()
    

gmm = mixture.GaussianMixture(n_components=n_classes, covariance_type='full').fit(X)

#colors = ['red', 'blue', 'green']
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture full')
for n, color in enumerate(colors):
    data = X[Y == n+1 ]
    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

gmm = mixture.GaussianMixture(n_components=n_classes, covariance_type='spherical').fit(X)

plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 1,
             'Gaussian Mixture spherical')

for n, color in enumerate(colors):
    data = X[Y == n + 1]
    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

gmm = mixture.GaussianMixture(n_components=n_classes, covariance_type='diag').fit(X)

plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 2,
             'Gaussian Mixture diag')
for n, color in enumerate(colors):
    data = X[Y == n + 1]
    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

gmm = mixture.GaussianMixture(n_components=n_classes, covariance_type='tied').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 3,
             'Gaussian Mixture tied')
for n, color in enumerate(colors):
    data = X[Y == n + 1]
    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)
plt.show()