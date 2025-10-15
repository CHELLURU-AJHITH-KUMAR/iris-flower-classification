#!/usr/bin/env python
# coding: utf-8

# Problem Statement:
# The Iris flower classification problem is a supervised machine learning task
# where the goal is to classify iris flowers into three species — Setosa, Versicolor, and Virginica —
# based on four features: sepal length, sepal width, petal length, and petal width.


    # Step 2: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Step 3: Load the Iris dataset
iris = load_iris()

# Convert to DataFrame for easier handling
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Show first 5 rows
df.head()

# Step 4: Exploratory Data Analysis (EDA)

# Check basic info
print(df.info())

# Check summary statistics
print(df.describe())

# Check class distribution
print(df['species'].value_counts())

# Pairplot to visualize relationships
sns.pairplot(df, hue='species')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

#Splits data into training and testing sets.
#Scales features so the model performs better.
#Trains a Logistic Regression classifier.
#Prints accuracy, precision/recall/F1, and confusion matrix.

# Step 5: Model Training

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split data into features and target
X = df.drop('species', axis=1)
y = df['species']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))




