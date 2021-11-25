#Importing Libraries

import sklearn
import numpy as np
import pandas as pd

#Linking to Google Drive

dataset = pd.read_csv('/content/sample_data/heart failur classification dataset.csv')
dataset.head()
dataset.shape

#Reading dataset

dataset = pd.read_csv('/content/sample_data/heart failur classification dataset.csv')
rows,cols = dataset.shape[0],dataset.shape[1]
dataset.head(7)
dataset.shape
dataset.isnull().sum()

#Handling missing values & Drop Empty columns

dataset = dataset.drop('Unnamed: 0', axis = 1)
dataset = dataset.dropna(how = 'any', axis = 0)
dataset.shape

#check Dataset

dataset.isnull().sum()

#Encoding Categorical Features

dataset.info()
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
dataset['sex'] = enc.fit_transform(dataset['sex'])
dataset['smoking'] = enc.fit_transform(dataset['smoking'])
print(dataset.info())
print(dataset.head(11))







#Scaling Using MinMax

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaler.fit(dataset)
md_scaled = scaler.transform(dataset)
print(md_scaled)

#Labelling And Training

from sklearn.model_selection import train_test_split
features = dataset[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']] 
label = dataset[['DEATH_EVENT']]
stratified = pd.DataFrame(label)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.25, stratify = stratified, random_state = 0)
print(X_train.shape)
print(X_test.shape)

#Features And Labelling

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#train
knn.fit(X_train_scaled, y_train)
#splitting the dataset into features and label
print("the features dataset after splitting is :")
print(features)
print("the label dataset after splitting is :")
print(label)








#Dataset Before Scaling

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.25, stratify = stratified, random_state = 0)
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
before_scale = knn.score(X_test, y_test)*100
print("Test set accuracy: {:.2f}".format(before_scale))

#Dataset After Scaling

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#train
knn.fit(X_train_scaled, y_train)
after_scale = knn.score(X_test_scaled,y_test)*100
# scoring on the scaled test set
#print("Scaled test set accuracy: {:.2f}".format(knn.score(X_test_scaled,y_test)))
print("Scaled test set accuracy: {:.2f}".format(after_scale))

#Bar Chart Of Before and After Scale

plt.bar(['After Scale', 'Before Scale'],[after_scale, before_scale])
plt.title('Comparing Accuracy')
plt.show()

