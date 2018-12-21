from models import *
from input_output import *

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.utils.class_weight import compute_sample_weight

## load .pickle, .csv and .txt files with data
x_train, y_train, x_test = load_files('all')

## extract k best features
#selector = SelectKBest(k=10)
#x_train = selector.fit_transform(x_train, y_train)

## split train set
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.1, random_state=50)

## models
#model = KNN()
#model = GaussNB()
#model = DecisionTree()

#model = LogisticSaga()
#model = LogisticSag()
#model = LogisticLbfgs()
#model = LogisticLiblinear()
#model = LogisticNewton()

#model = XGB1()
#model = XGB2()
#model = XGB3()
#model = XGB4()
#model = XGB5()
#model = XGB6()
#model = XGB7()

#model = XGBCustom1()
#model = XGBCustom2()
#model = XGBCustom3()
#model = XGBCustom4()

model = XGBFinal()


## find best value for parameters
#params = {'learning_rate':[i/1000.0 for i in range(75,86,1)]}
#model = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', n_jobs=-1, iid=False, cv=5)

## train model with weights
#s_weight = compute_sample_weight(class_weight='balanced', y=y_train)
#model.fit(x_train, y_train, sample_weight=s_weight)

## train model without weights
model.fit(x_train, y_train)

## print best parameters settings
#print(model.best_params_)
#print(model.best_score_)

## get data on train set
train_predictions = model.predict(x_validation)
print("Accuracy on train set: ", accuracy_score(y_validation, train_predictions))
#print("\nClassification report: \n", classification_report(y_validation, train_predictions))
#print("\nConfusion matrix: \n", confusion_matrix(y_validation, train_predictions))

## get results for test set
test_predictions = model.predict(x_test)

## export to csv file
export_to_csv(test_predictions)
