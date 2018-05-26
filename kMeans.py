from numpy import random, array
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float
from sklearn.model_selection import train_test_split
# evaluation methods
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import random


random.seed(42)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 
in_file = "Dataset.txt"
colnames = [ 'area A', 'perimeter P', 'compactness C = 4*pi*A/P^2','length of kernel','width of kernel','asymmetry coefficient', 'length of kernel groove', 'class']
wheatData = pd.read_csv(in_file,sep='\t', names = colnames);
#[row,col] = wheatData.shape
#print('Dataset has {0} columns and {1} rows'.format(col, row))
## 2. print top 5 lines
#print(wheatData.head(5))
df = wheatData.values
#
featureX = 0
featureY = 3
data = df[:,[featureX,featureY]]
kmeans = KMeans(n_clusters=3,random_state=5)

##split the dataset into training and validation sets using train test split(): half for training, and the other half for validation
featuresX = softmax(df[:,featureX]).reshape(-1, 1)
print('featuresX',featuresX)
featuresY = softmax(df[:,featureY]).reshape(-1, 1)
XtrainSet, XtestSet, YtrainSet, YtestSet = train_test_split(softmax(data[:,0]).reshape(-1, 1), softmax(data[:,1]).reshape(-1, 1),test_size=0.25)

print('XtrainSet',XtrainSet)
print('YtrainSet',YtrainSet)
kmeans.fit(scale(XtrainSet), scale(YtrainSet))
print('XtestSet',XtestSet)
print('YtestSet',YtestSet)
## make prediction using the testing sets
Ypred = kmeans.predict(XtestSet).reshape(-1, 1)
print('Ypred',Ypred)
#
##evaluate the performance of this model on the validation dataset by printing out the result of running classification_report()
print('Ypred.round()',np.around(Ypred, decimals=1))
accuracy = accuracy_score(YtestSet.round(), Ypred, normalize=False)
evaluation = classification_report(YtestSet.round(), Ypred)
print('Evaluation:',evaluation)
#accuracy = accuracy_score(YtestSet, Ypred)
r2_score = r2_score(YtestSet, Ypred)
print('The accuracy is:{0}. The r2_score is:{1}'.format(round(accuracy,2), round(r2_score,2)))


#kmeans.fit(scale(data))
labels = kmeans.predict(XtestSet)

#distcance from centroids
centroids = kmeans.cluster_centers_
print('centroids',centroids,'labels',labels)

#target = 7
#Y = df[:, target]
#X = np.array(df[:, [featureX, featureY]])
# 
#
##data = createClusteredData(100, 5)
#
#featuresX = df[:,featureX]
#featuresY = Y
##split the dataset into training and validation sets using train test split(): half for training, and the other half for validation
#XtrainSet, XtestSet, YtrainSet, YtestSet = train_test_split(featuresX , featuresY, test_size=0.75)
#
#
## machine learning algorithm, create a model by running it (fitting it ) on the training set
#model = KMeans(n_clusters=3)
## Note I'm scaling the data to normalize it! Important for good results
## train the model using the training sets
#print('XtrainSet',XtrainSet)
#print('XtestSet',XtestSet)
#print('YtrainSet',YtrainSet)
#model.fit(scale(XtrainSet),scale( YtrainSet))
#
## make prediction using the testing sets
#Ypred = model.predict(XtestSet)
#
##evaluate the performance of this model on the validation dataset by printing out the result of running classification_report()
#evaluation = classification_report(YtestSet, Ypred)
#print('Evaluation:',evaluation)
#accuracy = accuracy_score(YtestSet, Ypred)
#r2_score = r2_score(YtestSet, Ypred)
#print('The accuracy is:{0}. The r2_score is:{1}'.format(round(accuracy,2), round(accuracy,2)))
#
#
## We can look at the clusters each data point was assigned to
#print('model.labels_', model.labels_)
#
#
##dict={str(Kama), str(Rosa), str(Canadian)}
## And we'll visualize it:
#plt.figure(figsize=(8, 6))
#plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
#plt.title('kMeans scatter plot')
#plt.legend(scatterpoints = 10, loc ='lower right', prpp = dict(size =3)) # to do legend
#plt.xlabel(str(colnames[featureX]))
#plt.ylabel(str(colnames[featureY]))
#plt.show()
#colors = ['navy', 'turquoise', 'darkorange']
#for n, color in enumerate(colors):
#    data = X[Y == n+1]
#    plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)
#    plt.title('kMeans scatter plot')
#    plt.xlabel(str(colnames[featureX]))
#    plt.ylabel(str(colnames[featureY]))
