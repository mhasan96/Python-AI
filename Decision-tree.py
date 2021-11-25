
"""Importing Libraries"""
#Importing necessay libraries
#Importing necessay libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
# print(dataset.info())
# print(dataset.head(11))

"""Scaling Using MinMax"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaler.fit(dataset)
md_scaled = scaler.transform(dataset)
# print(md_scaled)
"""Labelling And Training"""

scaler = MinMaxScaler()
scaler.fit(dataset)
dataset_scaled = scaler.transform(dataset)
features = dataset[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']] 
label = dataset[['DEATH_EVENT']]
stratified = pd.DataFrame(label)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, stratify = stratified, random_state = 42)

"""Logistic Regression"""
#Train the model

model = LogisticRegression()
model.fit(x_train, y_train) #Training the model
predictions = model.predict(x_test)
#print(predictions)# printing predictions
logistric_regression=accuracy_score(y_test, predictions)
print("Before PCA Accuracy",(logistric_regression)*100,"%")
#print("Accuracy",(model.score(x_test,y_test)*100),"%")

"""Decision Tree"""

#Split the data into 80% training and 20% testing
model = DecisionTreeClassifier(criterion='entropy',random_state=1)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
decision_tree=accuracy_score(y_pred,y_test)
print("Before PCA Accuracy",(decision_tree)*100,"%")

#Support Vector Machine (SVM)
svm = SVC(kernel="linear")
#training
svm.fit(x_train, y_train.values.ravel())
predictionsSVM = svm.predict(x_test)
accuracySVM = accuracy_score(predictionsSVM, y_test)
print("Before PCA Accuracy",accuracySVM*100,"%")

# Neural Network
nnc = MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=1000)
#training
nnc.fit(x_train, y_train.values.ravel())
predictionsNNC = nnc.predict(x_test)
accuracyNNC = accuracy_score(predictionsNNC, y_test)
print("Before PCA Accuracy",accuracyNNC*100,"%")

#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 50)
#training
rfc.fit(x_train, y_train.values.ravel())
predictionsRFC = rfc.predict(x_test)
accuracyRFC = accuracy_score(predictionsRFC, y_test)
print("Before PCA Accuracy",accuracyRFC*100,"%")

#Bar Chart 
plt.bar(['Logistic\nRegression', 'Decision\nTree', 'Support\nVector\nMachine', 'Neural\nNetwork', 'Random\nForest'],[logistric_regression, decision_tree, accuracySVM, accuracyNNC, accuracyRFC])
plt.title('Without PCA Accuracy')
plt.show()

"""PCA

"""

columnCount = len(features.columns.values.tolist()) // 2
pca = PCA(n_components = columnCount)
principal_components = pca.fit_transform(features)
principalColummns = ["Principal Component" + str(i+1) for i in range(columnCount)]
principal_df = pd.DataFrame(data=principal_components, columns = principalColummns)
main_df=pd.concat([principal_df, dataset[["DEATH_EVENT"]]], axis=1)
main_df = main_df.dropna(how = 'any', axis = 0)

#pca Datasplit
pcaFeature = main_df.drop('DEATH_EVENT', axis = 1)
pcaLabel = main_df['DEATH_EVENT']
pcax_train, pcax_test, pcay_train, pcay_test = train_test_split(pcaFeature , pcaLabel, test_size=0.2, stratify = pcaLabel, random_state=0)

#Scaling
pcaLabelName = ['class']
pcaFeatureName = principalColummns
scaler = MinMaxScaler()
pcax_train = pd.DataFrame(scaler.fit_transform(pcax_train), columns = pcaFeatureName)
pcax_test = pd.DataFrame(scaler.fit_transform(pcax_test), columns = pcaFeatureName)

#Logistic Regression (After PCA)
model = LogisticRegression()
#training
model.fit(pcax_train, pcay_train)
predictions = model.predict(pcax_test)
accuracy = accuracy_score(pcay_test, predictions)
print("After PCA Accuracy",accuracy*100,"%")

#Decision Tree (After PCA)
model = DecisionTreeClassifier(criterion='entropy',random_state=1)
#training
model.fit(pcax_train,pcay_train)
prediction = model.predict(pcax_test)
accuracy = accuracy_score(prediction, pcay_test)
print("After PCA Accuracy",accuracy*100,"%")

#Support Vector Machine (After PCA)
svm = SVC(kernel="linear")
#training
svm.fit(pcax_train, pcay_train)
predictionsSVM = svm.predict(pcax_test)
pcaAccuracySupportVectorMachine = accuracy_score(predictionsSVM, pcay_test)
print("After PCA Accuracy",pcaAccuracySupportVectorMachine*100,"%")

#Neural Network (After PCA)
nnc = MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=1000)
#training
nnc.fit(pcax_train, pcay_train)
predictionsNNC = nnc.predict(pcax_test)
pcaAccuracyNeuralNetwork = accuracy_score(predictionsNNC, pcay_test)
print("After PCA Accuracy",pcaAccuracyNeuralNetwork*100,"%")

#Random Forest (After PCA)
rfc = RandomForestClassifier(n_estimators = 50)
#training
rfc.fit(pcax_train, pcay_train)
predictionsRFC = rfc.predict(pcax_test)
pcaAccuracyRandomForestClassifier = accuracy_score(predictionsRFC, pcay_test)
print("After PCA Accuracy",pcaAccuracyRandomForestClassifier*100,"%")

plt.bar(['Logistic\nRegression', 'Decision\nTree', 'Support\nVector\nMachine', 'Neural\nNetwork', 'Random\nForest'],[accuracy, accuracy, pcaAccuracySupportVectorMachine, pcaAccuracyNeuralNetwork, pcaAccuracyRandomForestClassifier])
plt.title('After PCA Accuracy')
plt.show()

#Accuracy Comparison
plt.bar(['P-LR','LR', 'P-DT','DT' ,'P-SVM','SVM', 'P-NN','NN', 'P-RF','RF'],[accuracy,logistric_regression, accuracy,decision_tree, pcaAccuracySupportVectorMachine,accuracySVM, pcaAccuracyNeuralNetwork,accuracyNNC, pcaAccuracyRandomForestClassifier,accuracyRFC])
plt.title('Accuracy Comparison')
plt.show()

