import pandas # CSV
import matplotlib.pyplot as plt #plot
import numpy as np #numpy

from sklearn import datasets
dataset=datasets.load_iris()

target=pandas.DataFrame(dataset.target)
target.columns=['targets']

dataset=pandas.DataFrame(dataset.data)
dataset.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# validation
X=dataset
Y=np.array(target).reshape(len(target),)
print(Y.shape)

#kmeans
from sklearn.cluster import KMeans

model = KMeans()
model.fit(X)

print(model.labels_)
