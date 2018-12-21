import pandas # CSV
import matplotlib.pyplot as plt #plot
import numpy as np #numpy

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import *

dataset=datasets.load_iris()

target=pandas.DataFrame(dataset.target)
target.columns=['targets']

dataset=pandas.DataFrame(dataset.data)
dataset.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

print(dataset.shape)
print(dataset.head(5))
print('')
print(dataset.describe())
print('')
print(target.groupby('targets').size())

X = dataset
Y = np.array(target).reshape(len(target),)

scoring = 'accuracy'
seed = 436
test_size = 0.2

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

kfold = KFold(n_splits=10, random_state=seed)

#cross_val_score(KNeighborsClassifier, x_train, x_test, TODO)

#model = KNeighborsClassifier()
#model.fit(X, Y)
#predictions = model.predict(X)

print("\nPrediction accuracy : ")
print(accuracy_score(Y, predictions))

