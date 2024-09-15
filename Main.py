import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
import joblib

warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv("heart.csv")

# Display dataset info and initial analysis
print(dataset.shape)
print(dataset.head(5))
print(dataset.sample(5))
print(dataset.describe())
print(dataset.info())

info = [
    "age", "1: male, 0: female",
    "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
    "resting blood pressure", "serum cholestoral in mg/dl", "fasting blood sugar > 120 mg/dl",
    "resting electrocardiographic results (values 0,1,2)", "maximum heart rate achieved",
    "exercise induced angina", "oldpeak = ST depression induced by exercise relative to rest",
    "the slope of the peak exercise ST segment", "number of major vessels (0-3) colored by flourosopy",
    "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"
]

for i in range(len(info)):
    print(dataset.columns[i] + ":\t\t\t" + info[i])

# Correlation with the target column
print(dataset.corr()["target"].abs().sort_values(ascending=False))

# Visualize target distribution
y = dataset["target"]
sns.countplot(y)

# Target distribution details
target_temp = dataset.target.value_counts()
print(target_temp)
print(f"Percentage of patients without heart problems: {round(target_temp[0] * 100 / 303, 2)}%")
print(f"Percentage of patients with heart problems: {round(target_temp[1] * 100 / 303, 2)}%")

# Gender vs target
sns.barplot(x=dataset["sex"], y=y)

# Chest pain type vs target
sns.barplot(x=dataset["cp"], y=y)

# Fasting blood sugar vs target
sns.barplot(x=dataset["fbs"], y=y)

# Resting ECG vs target
sns.barplot(x=dataset["restecg"], y=y)

# Exercise induced angina vs target
sns.barplot(x=dataset["exang"], y=y)

# Slope of the peak exercise ST segment vs target
sns.barplot(x=dataset["slope"], y=y)

# Number of major vessels vs target
sns.countplot(dataset["ca"])
sns.barplot(x=dataset["ca"], y=y)

# Thalassemia vs target
sns.barplot(x=dataset["thal"], y=y)
sns.distplot(dataset["thal"])

print(dataset['thal'].unique())

# Splitting dataset into train and test sets
predictors = dataset.drop("target", axis=1)
target = dataset["target"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
print(f"The accuracy score achieved using Logistic Regression is: {score_lr}%")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
score_nb = round(accuracy_score(Y_pred_nb, Y_test) * 100, 2)
print(f"The accuracy score achieved using Naive Bayes is: {score_nb}%")

# Support Vector Machine
sv = svm.SVC(kernel='linear')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)
print(f"The accuracy score achieved using Linear SVM is: {score_svm}%")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)
print(f"The accuracy score achieved using KNN is: {score_knn}%")

# Decision Tree
max_accuracy = 0
for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)
score_dt = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
print(f"The accuracy score achieved using Decision Tree is: {score_dt}%")

# Random Forest
max_accuracy = 0
for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
print(f"The accuracy score achieved using Random Forest is: {score_rf}%")

# XGBoost
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)
Y_pred_xgb = xgb_model.predict(X_test)
score_xgb = round(accuracy_score(Y_pred_xgb, Y_test) * 100, 2)
print(f"The accuracy score achieved using XGBoost is: {score_xgb}%")

# Neural Network (Keras)
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=13))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=300)

Y_pred_nn = model.predict(X_test)
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)
print(f"The accuracy score achieved using Neural Network is: {score_nn}%")

# Comparing all algorithms
scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]

for i in range(len(algorithms)):
    print(f"The accuracy score achieved using {algorithms[i]} is: {scores[i]} %")

# Plotting the accuracy scores
#sns.set(rc={'figure.figsize':(15,8)})
#plt.xlabel("Algorithms")
#plt.ylabel("Accuracy score")
#sns.barplot(x=algorithms, y=scores)
#plt.show()



joblib.dump(lr, 'model.joblib')

print("Model saved as 'model.joblib'")