"""Importing Libraries"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

"""Linking too Google Drive """

dataset = pd.read_csv('/content/sample_data/heart failur classification dataset.csv')
dataset.head()

"""Preprocessing dataset"""

dataset = pd.read_csv('/content/sample_data/heart failur classification dataset.csv')
rows,cols = dataset.shape[0],dataset.shape[1]
dataset.head(7)
dataset.shape
dataset.isnull().sum()
dataset = dataset.drop('Unnamed: 0', axis = 1)
dataset = dataset.dropna(how = 'any', axis = 0)
dataset.info()	

#encoding categorical feature
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
dataset['sex'] = enc.fit_transform(dataset['sex'])
dataset['smoking'] = enc.fit_transform(dataset['smoking'])
print(dataset.info())
print(dataset.head(11))

"""Scaling Using MinMax"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaler.fit(dataset)
md_scaled = scaler.transform(dataset)
print(md_scaled)
"""Labelling And Training"""

from sklearn.model_selection import train_test_split
features = dataset[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']] 
label = dataset[['DEATH_EVENT']]
stratified = pd.DataFrame(label)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, stratify = stratified, random_state = 0)
print(X_train.shape)
print(X_test.shape)

"""Seperating features and labels"""

#Prepare the training set
# X = feature values, all the columns except the last column
X = dataset.iloc[:, :-1]
# y = target values, last column of the data frame
y = dataset.iloc[:, -1]
#Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

"""Logistic Regression"""

#Train the model
model = LogisticRegression()
model.fit(x_train, y_train) #Training the model
predictions = model.predict(x_test)
#print(predictions)# printing predictions
score1=accuracy_score(y_test, predictions)
print("Accuracy",(score1)*100,"%")
#print("Accuracy",(model.score(x_test,y_test)*100),"%")

"""Decision Tree"""

#Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
model = DecisionTreeClassifier(criterion='entropy',random_state=1)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
score=accuracy_score(y_pred,y_test)
print("Accuracy",(score)*100,"%")

"""Scatter Chart Of Logistic and Regression"""

import matplotlib.pyplot as plt
plt.bar(['Logistic Regression', 'Decision Tree'],[score1, score])
plt.title('Comparing Accuracy')
plt.show()

