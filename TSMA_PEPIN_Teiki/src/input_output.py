import pickle
import numpy as np
import pandas as pd
import random

###################### LOAD FILES ######################
def load_files(pickle_chosen):
  # load corrupted files ids
  corrupted_train, corrupted_test = load_corrupted_ids()

  # display number of corrupted files
  nb_corr = len(corrupted_train) + len(corrupted_test)
  print(str(nb_corr) + " corrupted files detected.")

  # load pickle data
  if pickle_chosen == 'all':
    x_train, x_test = load_pickles_all()
  elif pickle_chosen == 'extra':
    x_train, x_test = load_pickles_extra()
  elif pickle_chosen == 'mfcc':
    x_train, x_test = load_pickles_mfcc()
  else:
    x_train, x_test = load_pickles_all()	  

  # load csv files data
  _, y_train = load_train_csv(corrupted_train)
  
  return x_train, y_train, x_test;
  
################### LOAD PICKLE MFCC ######################
def load_pickles_mfcc():
  x_train = load_pickle_mfcc('../data/pickle_train_mfcc.pickle')
  x_test  = load_pickle_mfcc('../data/pickle_test_mfcc.pickle')
  
  return x_train, x_test;
  
def load_pickle_mfcc(pathname):
  pickle_file = open(pathname, 'rb')
  file_data = pickle.load(pickle_file)
  pickle_file.close()
  
  features = []
  for f in file_data:
    features.append(np.concatenate((f[1][0], f[1][1]), axis=None))

  features = pd.DataFrame(features)
  
  return features
  
################### LOAD PICKLE EXTRA ######################
def load_pickles_extra():
  x_train = load_pickle_extra('../data/pickle_train_extra.pickle')
  x_test  = load_pickle_extra('../data/pickle_test_extra.pickle')
  
  return x_train, x_test;
  
def load_pickle_extra(pathname):
  pickle_file = open(pathname, 'rb')
  file_data = pickle.load(pickle_file)
  pickle_file.close()
  
  features = []
  for f in file_data:
    features.append(np.concatenate((f[1][0], f[1][1], f[2][0], f[2][1], f[3][0], f[3][1], f[4][0], f[4][1], f[5][0], f[5][1], f[6][0], f[6][1], f[7][0], f[7][1]), axis=None))

  features = pd.DataFrame(features)
  
  return features
  
################### LOAD PICKLE ALL ######################
def load_pickles_all():
  x_train = load_pickle_all('../data/pickle_train_all.pickle')
  x_test  = load_pickle_all('../data/pickle_test_all.pickle')
  
  return x_train, x_test;
  
def load_pickle_all(pathname):
  pickle_file = open(pathname, 'rb')
  file_data = pickle.load(pickle_file)
  pickle_file.close()
  
  features = []
  for f in file_data:
    features.append(np.concatenate((f[1][0], f[1][1], f[2][0], f[2][1], f[3][0], f[3][1], f[4][0], f[4][1], f[5][0], f[5][1], f[6][0], f[6][1], f[7][0], f[7][1], f[8][0], f[8][1], f[9][0], f[9][1], f[10][0], f[10][1], f[11][0], f[11][1], f[12][0], f[12][1], f[13][0], f[13][1]), axis=None))

  features = pd.DataFrame(features)
  
  return features
  
################### LOAD CORRUPTED ######################
def load_corrupted_ids():
  corrupted_train = []
  with open('../data/corrupted_files_train.txt', 'r') as corr_file:
    for line in corr_file:
      corrupted_train.append(int(line[:-1]))

  corrupted_test = []
  with open('../data/corrupted_files_test.txt', 'r') as corr_file:
    for line in corr_file:
      corrupted_test.append(int(line[:-1]))
      
  return corrupted_train, corrupted_test;
  
###################### LOAD CSV #########################
def load_train_csv(corrupted):
  data = pd.read_csv('../data/train.csv').values

  # delete corrupted data
  for c in corrupted:
    data = np.delete(data, np.argwhere(data[:, 0] == c), 0)

  labels = data[:, 0]
  genres = data[:, 1]
  
  return labels, genres;
  
def load_test_csv(corrupted):
  data = pd.read_csv('../data/test.csv').values

  labels = data[:, 0]
  
  return labels
  
  
################ EXPORT TO CSV ######################
def export_to_csv(predictions):
  _, corrupted_test = load_corrupted_ids()
  
  labels_test = load_test_csv(corrupted_test)
  
  for c in corrupted_test:
    ind_corrupted =  int(np.argwhere(labels_test == c))
    predictions = np.insert(predictions, ind_corrupted, random.randint(1,8))

  results = pd.DataFrame(data = {'track_id': labels_test, 'genre_id': predictions}, columns = ['track_id', 'genre_id'])
  
  results.to_csv("../data/results.csv", index=False)
  
  print("Results for test set exported to results.csv")
  
  return;
