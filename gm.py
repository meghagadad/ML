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

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
# READ DB
colnames = ['area A', 'perimeter P', 'compactness C = 4*pi*A/P^2','length of kernel','width of kernel','asymmetry coefficient', 'length of kernel groove', 'class']
wheatData = pd.read_csv(in_file,sep='\t', names = colnames);


# give me to features
dF = wheatData.values
featureA = 0
featureB = 3
data = dF[:,[featureA,featureB ]]
[row,col] = data.shape
# Generate random sample, two components
np.random.seed(0)
SampleList=[]
SampleArray = np.array([])
for i in range(2):
    randomRow = np.random.choice(row,1)
    C =  np.array(dF[int(randomRow),[featureA,featureB ]])
    SampleArray = np.append(SampleArray, C)
    np.array(SampleList.append(C))
    print('C ',C, i , type(C))
print('SampleList ',SampleList, type(SampleList) )

  
X = np.array(dF[:,[featureA,featureB ]] )
print('x', X, type(X))
#df.drop([randomRow])

print('min',min(dF[:,featureA]))
print('max',max(dF[:,featureA]))
print('min',min(dF[:,featureB]))
print('max',max(dF[:,featureB]))
# Number of samples per component
n_samples = 100

# Generate random sample, two components
##
#C = np.array([[0., -0.1], [1.7, .4]])
#print('C ',C, type(C))
#
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
#print('x', X, type(X))

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')

# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                        covariance_type='full').fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian Gaussian Mixture with a Dirichlet process prior')

plt.show()





    
    


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(10., 22.)
    plt.ylim(4., 7.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
