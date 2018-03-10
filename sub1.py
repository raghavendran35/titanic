import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

import warnings
warnings.filterwarnings('ignore')

#take care of missing data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, Imputer
from sklearn.svm import SVC
train = pd.read_csv('train.csv')
##To create actual model and do cross validation testing:
# take columns Sex, Age, SibSp, Fare, Embarked, Parch
# don't worry about pclass b/c that will be reflected in Fare
# feature scale for non-categorical variables
train['Age'] = (train['Age']-train['Age'].mean())/(train['Age'].std()) 
train['SibSp'] = (train['SibSp']-train['SibSp'].mean())/(train['SibSp'].std()) 
train['Fare'] = (train['Fare']-train['Fare'].mean())/(train['Fare'].std()) 
train['Parch'] = (train['Parch']-train['Parch'].mean())/(train['Parch'].std())
train['Age'] = train['Age'].fillna(train['Age'].mean())
#one hot encode all the categorical variables: 
#Sex, remember that more females were likely to survive
#Embarked, not actually that significant, but still noteworthy
cleanup_e = {'Embarked': {'S': train['Embarked'].value_counts('S')['S'], 'C' : train['Embarked'].value_counts('C')['C'], 'Q': train['Embarked'].value_counts('Q')['Q']}} 
train.replace(cleanup_e, inplace = True)
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mean())
cleanup_s = {'Sex':{'male': train['Sex'].value_counts('male')['male'], 'female': train['Sex'].value_counts('male')['female']} }
train.replace(cleanup_s, inplace = True)

obtained = train.values.tolist()
y = [x[1] for x in obtained]
X = []
for x in obtained:
	current_list = [round(i,4) for i in x[4:8]]
	current_list += [round(x[9], 4), round(x[11], 4)]
	X.append(current_list)



plot_z = []
plot_c = []
plot_d = []


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
C = [.0001, .001, .01, .1, 1, 10, 100]
d = [1,2,3,4,5]
for i in d:
    for c in C:
        plot_d.append(i)
        plot_c.append(c)
        s = SVC(C = c, kernel = 'poly', degree = i, probability = True)
        clf = s.fit(X_train, y_train)
        plot_z.append(clf.score(X_test, y_test))
index = [i for i in range(len(plot_z)) if plot_z[i] == max(plot_z)][0]

best_degree = plot_d[index]
best_C = plot_c[index]
s = SVC(C = best_C, kernel = 'poly', degree = best_degree, probability = True)
c = s.fit(X_train, y_train)

test = pd.read_csv('test.csv')
#test variables are 
#Age, SibSp, Fare, Embarked, Parch, Embarked, Sex
test['Age'] = (test['Age']-test['Age'].mean())/(test['Age'].std()) 
test['SibSp'] = (test['SibSp']-test['SibSp'].mean())/(test['SibSp'].std()) 
test['Fare'] = (test['Fare']-test['Fare'].mean())/(test['Fare'].std()) 
test['Parch'] = (test['Parch']-test['Parch'].mean())/(test['Parch'].std())
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
cleanup_E = {'Embarked': {'S': test['Embarked'].value_counts('S')['S'], 'C' : test['Embarked'].value_counts('C')['C'], 'Q': test['Embarked'].value_counts('Q')['Q']}} 
test.replace(cleanup_E, inplace = True)
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mean())
cleanup_S = {'Sex':{'male': test['Sex'].value_counts('male')['male'], 'female': test['Sex'].value_counts('male')['female']} }
test.replace(cleanup_S, inplace = True)
ot = test.values.tolist()
x_test = []
for x in ot:
	current_list = [round(i,4) for i in x[3:7]]
	current_list += [round(x[8], 4), round(x[10], 4)]
	x_test.append(current_list)
test_results = ['Survived'] + c.predict(x_test).tolist()

ids = ['PassengerId'] + test['PassengerId'].tolist()
import csv
file1 = open('final.csv', 'w')
for index in range(len(ids)):
	file1.write(str(ids[index]) + "," + str(test_results[index]) + "\n")
file1.close()

