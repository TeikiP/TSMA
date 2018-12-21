from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier



def KNN():
  return KNeighborsClassifier()
  
def GaussNB():
  return GaussianNB()
  
def DecisionTree():
  return DecisionTreeClassifier()
  
  
  
def LogisticSaga():
  return LogisticRegression(solver='saga', multi_class='auto')
  
def LogisticSag():
  return LogisticRegression(solver='sag', multi_class='auto')
  
def LogisticLbfgs():
  return LogisticRegression(solver='lbfgs', multi_class='auto')
  
def LogisticLiblinear():
  return LogisticRegression(solver='liblinear', multi_class='auto')
  
def LogisticNewton():
  return LogisticRegression(solver='newton-cg', multi_class='auto')




def XGB1():
  return XGBClassifier(n_jobs=-1, random_state=50)

def XGB2():
  return XGBClassifier(n_jobs=-1, random_state=50, objective='multi:softmax', num_class=8)

def XGB3():
  return XGBClassifier(n_jobs=-1, random_state=50, objective='multi:softmax', num_class=8, min_child_weight=6, max_depth=8)
  
def XGB4():
  return XGBClassifier(n_jobs=-1, random_state=50, objective='multi:softmax', num_class=8, min_child_weight=6, max_depth=8, gamma=0.3)
  
def XGB5():
  return XGBClassifier(n_jobs=-1, random_state=50, objective='multi:softmax', num_class=8, min_child_weight=6, max_depth=8, gamma=0.3, subsample=0.8, colsample_bytree=0.8)
  
def XGB6():
  return XGBClassifier(n_jobs=-1, random_state=50, objective='multi:softmax', num_class=8, min_child_weight=6, max_depth=8, gamma=0.3, subsample=0.8, colsample_bytree=0.8, learning_rate=0.1, booster='gbtree')
  
def XGB7():
  return XGBClassifier(n_jobs=-1, random_state=50, objective='multi:softmax', num_class=8, min_child_weight=6, max_depth=8, gamma=0.3, subsample=0.8, colsample_bytree=0.8, learning_rate=0.1, booster='gbtree', nb_estimators=300)




def XGBCustom1():
  return XGBClassifier(learning_rate = 0.1, n_estimators = 1000, n_jobs=-1, max_depth=5, colsample_bytree=0.8, objective='multi:softmax', random_state=50)

def XGBCustom2():
  return XGBClassifier(learning_rate = 0.1, n_estimators = 1500, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='multi:softmax', n_jobs=-1, scale_pos_weight=1, random_state=50)
  
def XGBCustom3():
  return XGBClassifier(learning_rate = 0.05, n_estimators = 300, n_jobs=-1, max_depth=3, random_state=50)
  
def XGBCustom4():
  return XGBClassifier(n_jobs=-1, random_state=50, objective='multi:softmax', num_class=8, min_child_weight=1, max_depth=5, gamma=0, subsample=0.8, colsample_bytree=0.8, learning_rate=0.15, booster='gbtree', n_estimators=1000, scale_pos_weight=1)
  
  

  
def XGBFinal():
  return XGBClassifier(n_jobs=-1, random_state=50, objective='multi:softmax', num_class=8, min_child_weight=1, max_depth=5, gamma=0, subsample=0.8, colsample_bytree=0.8, learning_rate=0.080, booster='gbtree', n_estimators=542, scale_pos_weight=1)

