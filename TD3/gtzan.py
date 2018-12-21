import librosa, librosa.display
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas

# verify / load
picklefile=open('gtzan_pickle.pickle', 'rb')
ds = pickle.load(picklefile)
picklefile.close()

labels=[]
x=[]

for d in ds:
    labels.append(d[0])
    x.append(np.concatenate((d[1][0],d[1][1])))

#
dataset=pandas.DataFrame(x)

print(dataset.shape)
print(dataset.head(5))

# stats
print(dataset.describe())
