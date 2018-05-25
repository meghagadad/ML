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
from sklearn import datasets

from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn import mixture

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

    plt.xlim(min(dF[:, featureA]), max(dF[:, featureA]))
    plt.ylim(min(dF[:, featureB]), max(dF[:, featureB]))
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


color_iter = itertools.cycle(['yellow', 'pink', 'purple'])

in_file = 'seeds_dataset.txt'
colnames = ['area A', 'perimeter P', 'compactness C = 4*pi*A/P^2', 'length of kernel', 'width of kernel',
            'asymmetry coefficient', 'length of kernel groove', 'class']
wheatData = pd.read_csv(in_file, delim_whitespace=True, names=colnames);

dF = wheatData.values
featureA = 1
featureB = 5
target = 7
data = dF[:, [featureA, featureB]]
Y = dF[:, target]
[row, col] = data.shape
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

print('min', min(dF[:, featureA]))
print('max', max(dF[:, featureA]))
print('min', min(dF[:, featureB]))
print('max', max(dF[:, featureB]))

n_samples = 210

n_classes = 3
gmm = mixture.GaussianMixture(n_components=n_classes, covariance_type='full').fit(X)

colors = ['red', 'blue', 'green']
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture full')
for n, color in enumerate(colors):
    data = X[Y == n + 1]
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


#THIS SHIT IS NEW!



iris = datasets.load_iris()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(iris.data, iris.target)))


X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
                   covariance_type=cov_type, max_iter=20, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    plot_results(estimator, h)
    
    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
                    label=iris.target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(titlethingy)

plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))


plt.show()