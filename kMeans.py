from numpy import random, array
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
# evaluation methods
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

in_file = "Dataset.txt"
colnames = [ 'area A', 'perimeter P', 'compactness C = 4*pi*A/P^2','length of kernel','width of kernel','asymmetry coefficient', 'length of kernel groove', 'class']
wheatData = pd.read_csv(in_file,delim_whitespace =True, names = colnames);
df = wheatData.values
featureX = 1
featureY = 6
data = df[:,[featureX,featureY]]

X = (np.array(dF[:, [featureA, featureB]]))
target = 7
Y = dF[:, target]
# create a model using KMeans
kmeans = KMeans(n_clusters=3,random_state=5)

##split the dataset into training and validation sets using train test split()
featuresX = scale(df[:,featureX]).reshape(-1, 1)
featuresY = scale(df[:,featureY]).reshape(-1, 1)
XtrainSet, XtestSet, YtrainSet, YtestSet = train_test_split(df, Y,test_size=0.25)

#train model
kmeans.fit(XtrainSet,YtrainSet)

## make prediction using the testing sets
Ypred = kmeans.predict(XtestSet).reshape(-1, 1)

##evaluate the performance of this model on the validation dataset by printing out the result of running classification_report()
evaluation = classification_report(YtestSet.round(), Ypred)
print('Evaluation:',evaluation)
accuracy = accuracy_score(YtestSet.round(), Ypred, normalize=False)
r2_score = r2_score(YtestSet, Ypred)
print('The accuracy is:{0}. The r2_score is:{1}'.format(round(accuracy,2), round(r2_score,2)))

#distcance from centroids
centroids = kmeans.cluster_centers_
print('centroids',centroids)

target = 7
Y = df[:, target]
X = np.array(df[:, [featureX, featureY]])


colors = ['navy', 'turquoise', 'darkorange']
for n, color in enumerate(colors):
    data = X[Y == n+1]
    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)
    plt.title('kMeans scatter plot')
    plt.xlabel(str(colnames[featureX]))
    plt.ylabel(str(colnames[featureY]))
plt.show()
