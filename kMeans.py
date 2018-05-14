from numpy import random, array





from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float



in_file = "Dataset.txt"
colnames = [ 'area A', 'perimeter P', 'compactness C = 4*pi*A/P^2','length of kernel','width of kernel','asymmetry coefficient', 'length of kernel groove', 'class']
wheatData = pd.read_csv(in_file,sep='\t', names = colnames);
[row,col] = wheatData.shape
print('Dataset has {0} columns and {1} rows'.format(col, row))
# 2. print top 5 lines
print(wheatData.head(5))
df = wheatData.values
   
data = df[:,[2,7]]

 

#data = createClusteredData(100, 5)

model = KMeans(n_clusters=3)

# Note I'm scaling the data to normalize it! Important for good results.
model = model.fit(scale(data))

# We can look at the clusters each data point was assigned to
print(model.labels_)

# And we'll visualize it:
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
plt.show()