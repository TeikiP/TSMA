import pickle
import numpy as np
import pandas as pd

picklefile = open('../data/pickle_train_all.pickle', 'rb')
songs = pickle.load(picklefile)
picklefile.close()

labels = []
x = []

for s in songs:
    labels.append(s[0])
    x.append(np.concatenate((s[1][0], s[1][1], s[2][0], s[2][1], s[3][0], s[3][1], s[4][0], s[4][1], s[5][0], s[5][1], s[6][0], s[6][1], s[7][0], s[7][1], s[8][0], s[8][1], s[9][0], s[9][1], s[10][0], s[10][1], s[11][0], s[11][1], s[12][0], s[12][1], s[13][0], s[13][1]), axis=None))

dataset = pd.DataFrame(x)

print(dataset.shape)
print(dataset.head(5))
print(dataset.describe())

print('end')
