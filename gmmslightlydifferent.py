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

from sklearn import mixture

# clusterColors = ["orange", "purple", "pink"]

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

        #        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(min(dF[:, featureA]), max(dF[:, featureA]))
    plt.ylim(min(dF[:, featureB]), max(dF[:, featureB]))
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


color_iter = itertools.cycle(['yellow', 'pink', 'purple'])
# READ DB
in_file = 'buggy_seeds.txt'
colnames = ['area A', 'perimeter P', 'compactness C = 4*pi*A/P^2', 'length of kernel', 'width of kernel',
            'asymmetry coefficient', 'length of kernel groove', 'class']
wheatData = pd.read_csv(in_file, delim_whitespace=True, names=colnames);

# give me to features
dF = wheatData.values
featureA = 0
featureB = 3
target = 7
data = dF[:, [featureA, featureB]]
Y = dF[:, target]
[row, col] = data.shape
# Generate random sample, two components
np.random.seed(0)
SampleList = []
SampleArray = np.array([])
for i in range(2):
    randomRow = np.random.choice(row, 1)
    C = np.array(dF[int(randomRow), [featureA, featureB]])
    SampleArray = np.append(SampleArray, C)
    np.array(SampleList.append(C))
    print('C ', C, i, type(C))
print('SampleList ', SampleList, type(SampleList))

X = np.array(dF[:, [featureA, featureB]])
print('x', X, type(X))
# df.drop([randomRow])

print('min', min(dF[:, featureA]))
print('max', max(dF[:, featureA]))
print('min', min(dF[:, featureB]))
print('max', max(dF[:, featureB]))
# Number of samples per component
n_samples = 210

# Generate random sample, two components
##
# C = np.array([[0., -0.1], [1.7, .4]])
# print('C ',C, type(C))
#
# X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
# print('x', X, type(X))

# Fit a Gaussian mixture with EM using five components
n_classes = 3
gmm = mixture.GaussianMixture(n_components=n_classes, covariance_type='full').fit(X)
# Plot the test data with crosses
colors = ['red', 'blue', 'green']
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture full')
for n, color in enumerate(colors):
    data = X[Y == n + 1]
    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

gmm = mixture.GaussianMixture(n_components=n_classes, covariance_type='spherical').fit(X)
# draw clusters
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 1,
             'Gaussian Mixture spherical')
# draw samples
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

# Fit a Dirichlet process Gaussian mixture using five components
# dpgmm = mixture.BayesianGaussianMixture(n_components=n_classes,
#                                        covariance_type='full').fit(X)
#
### Try GMMs using different types of covariances.
##estimators = dict((cov_type, mixture.GaussianMixture(n_components=n_classes,
##                   covariance_type=cov_type, max_iter=20, random_state=0))
##                  for cov_type in ['spherical', 'diag', 'tied', 'full'])
##
##n_estimators = len(estimators)
##
##plt.figure(figsize=(3 * n_estimators // 2, 6))
##plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
##                    left=.01, right=.99)
#
#
#
#
# plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 3,
#             'Bayesian Gaussian Mixture with a Dirichlet process prior')

plt.show()