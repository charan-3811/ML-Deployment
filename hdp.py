# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ICyETJ9mpTNK_bp0BBIzqHDPp6cHvD-b

HEART DISEASE PREDICTION
"""

import pandas as pd
df = pd.read_csv("heart.csv")
df.head(5)

"""CHECKING DUPLICATES"""

data_dup = df.duplicated().any()

df = df.drop_duplicates()
data_dup = df.duplicated().any()

df.isnull().sum()

df.info()

import matplotlib.pyplot as plt
import seaborn as sns
data = df
categorical_cols = ['age','sex', 'cp','trestbps', 'fbs', 'restecg', 'thalach','exang','oldpeak', 'slope', 'ca', 'thal']
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=col, hue='target')
    plt.title(f'Countplot of {col} by Target')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(['No Heart Disease', 'Heart Disease'])
    plt.show()

"""SPLITTNG DATA INTO TRIAING AND TESTING"""

heart_data=df
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

X_cls = heart_data.drop('target', axis=1)
y_cls = heart_data['target']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

"""KNN"""

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_cls, y_train_cls)
y_pred_knn_cls = knn_classifier.predict(X_test_cls)
accuracy_knn_cls = accuracy_score(y_test_cls, y_pred_knn_cls)

"""SVM"""

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_cls, y_train_cls)
y_pred_svm_cls = svm_classifier.predict(X_test_cls)
accuracy_svm_cls = accuracy_score(y_test_cls, y_pred_svm_cls)

"""DECISION TREE"""

tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train_cls, y_train_cls)
y_pred_tree_cls = tree_classifier.predict(X_test_cls)
accuracy_tree_cls = accuracy_score(y_test_cls, y_pred_tree_cls)

"""RANDOM FOREST"""

rf_classifier = RandomForestClassifier(n_estimators=50)
rf_classifier.fit(X_train_cls, y_train_cls)
y_pred_rf_cls = rf_classifier.predict(X_test_cls)
accuracy_rf_cls = accuracy_score(y_test_cls, y_pred_rf_cls)

"""NAVIE BIAS"""

nb_classifier = GaussianNB()
nb_classifier.fit(X_train_cls, y_train_cls)
y_pred_nb_cls = nb_classifier.predict(X_test_cls)
accuracy_nb_cls = accuracy_score(y_test_cls, y_pred_nb_cls)

X_reg = heart_data.drop('target', axis=1)
y_reg = heart_data['target']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
linear_regression = LinearRegression()
linear_regression.fit(X_train_reg, y_train_reg)
y_pred_reg = linear_regression.predict(X_test_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)

"""PRINTING ALL THE RESULTS

"""

print("Classification Results:")
print("K-Nearest Neighbors (KNN) Accuracy:", accuracy_knn_cls)
print("Support Vector Machine (SVM) Accuracy:", accuracy_svm_cls)
print("Decision Tree Accuracy:", accuracy_tree_cls)
print("Random Forest Accuracy:", accuracy_rf_cls)
print("Naive Bayes Accuracy:", accuracy_nb_cls)

print("\nRegression Results:")
print("Linear Regression Mean Squared Error:", mse_reg)

import pickle

filename = 'heart-disease.pkl'
pickle.dump(rf_classifier, open(filename, 'wb'))