from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import float
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# evaluation methods
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

in_file = "Dataset.txt"
colnames = [ 'area A', 'perimeter P', 'compactness C = 4*pi*A/P^2','length of kernel','width of kernel','asymmetry coefficient', 'length of kernel groove', 'class']
wheatData = pd.read_csv(in_file,sep='\t', names = colnames);
[row,col] = wheatData.shape
print('Dataset has {0} columns and {1} rows'.format(col, row))
# 2. print top 5 lines
print(wheatData.head(5))
dF = wheatData.values

featureX = 0
featureY = 3
data = dF[:,[featureX,featureY]]
target = 7
Y = dF[:, target]
X = np.array(dF[:, [featureX, featureY]])


# machine learning algorithm, create a model by running it (fitting it ) on the training set
model = KMeans(n_clusters=3,init='random', max_iter=100, n_init=1, verbose=1)
featuresX = df[:,featureX]
featuresY = df[:, featureY]
#split the dataset into training and validation sets using train test split(): half for training, and the other half for validation
XtrainSet, XtestSet, YtrainSet, YtestSet = train_test_split(featuresX , Y, test_size=0.75)
# Note I'm scaling the data to normalize it! Important for good results
# train the model using the training sets
XtrainSet = XtrainSet.reshape(-1, 1)
YtrainSet =  YtrainSet.reshape(-1, 1)
XtestSet= XtestSet.reshape(-1, 1)
YtestSet= YtestSet.reshape(-1, 1)

model.fit(XtrainSet, YtrainSet)
print('model.labels_', model.labels_)

print('XtrainSet',XtestSet)

print('YtrainSet',YtrainSet)
# make prediction using the testing sets
Ypred = model.predict(XtestSet)
print('YtestSet',YtestSet)
#evaluate the performance of this model on the validation dataset by printing out the result of running classification_report()
evaluation = classification_report(YtestSet, Ypred)
print('Evaluation:',evaluation)
accuracy = accuracy_score(YtestSet, Ypred)
r2_score = r2_score(YtestSet, Ypred)
print('The accuracy is:{0}. The r2_score is:{1}'.format(round(accuracy,2), round(accuracy,2)))



# Note I'm scaling the data to normalize it! Important for good results.
#model = model.fit(scale(data))

# We can look at the clusters each data point was assigned to
print('model.labels_', model.labels_)


#dict={str(Kama), str(Rosa), str(Canadian)}
# And we'll visualize it:
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1])#, c=model.labels_.astype(float))
plt.title('kMeans scatter plot')
plt.legend(scatterpoints = 10, loc ='lower right', prpp = dict(size =3)) # to do legend
plt.xlabel(str(colnames[featureX]))
plt.ylabel(str(colnames[featureY]))
plt.show()
colors = ['navy', 'turquoise', 'darkorange']
for n, color in enumerate(colors):
    data = X[Y == n+1]
    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)
    plt.title('kMeans scatter plot')
    plt.xlabel(str(colnames[featureX]))
plt.ylabel(str(colnames[featureY]))



