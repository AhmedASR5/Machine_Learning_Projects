import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

#lodading the data
url = 'processed.cleveland.data' # URL for the Heart Disease dataset
column_names = [
    "Age (years)",
    "Sex (1: Male, 0: Female)",
    "Chest Pain Type (1-4)",
    "Resting Blood Pressure (mm Hg)",
    "Cholesterol (mg/dl)",
    "Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)",
    "Resting ECG Results (0-2)",
    "Max Heart Rate (beats/min)",
    "Exercise-Induced Angina (1: Yes, 0: No)",
    "ST Depression (induced by exercise relative to rest)",
    "Slope of Peak Exercise ST Segment (1-3)",
    "Number of Major Vessels (0-3) colored by fluoroscopy",
    "Thalassemia (3: Normal, 6: Fixed Defect, 7: Reversible Defect)",
    "           Heart Disease Presence (1: Present, 0: Absent)"
]
heart_disease_data = pd.read_csv(url, names=column_names, na_values="?")

# Convert 'Heart Disease Presence' to a binary classification task and handle missing values
heart_disease_data['           Heart Disease Presence (1: Present, 0: Absent)'] = heart_disease_data['           Heart Disease Presence (1: Present, 0: Absent)'].apply(lambda x: 1 if x > 0 else 0)
heart_disease_data.fillna(heart_disease_data.median(), inplace=True)

# Basic descriptive statistics
print(heart_disease_data.describe())

# Visualizations
plt.figure(figsize=(15, 10))
for i, column in enumerate(heart_disease_data.columns):
    plt.subplot(4, 4, i + 1)
    sns.histplot(heart_disease_data[column], kde=True)
    plt.xlabel(column)  # Setting x-axis label to the column name for clarity
    plt.ylabel("Number of Patients")  # Setting y-axis label
plt.tight_layout()
plt.show()

# Data preparation
X = heart_disease_data.drop('           Heart Disease Presence (1: Present, 0: Absent)', axis=1)
y = heart_disease_data['           Heart Disease Presence (1: Present, 0: Absent)']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate kNN model
def train_evaluate_knn(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Evaluating kNN models
accuracy_k1 = train_evaluate_knn(1)
accuracy_k3 = train_evaluate_knn(3)
print(f'kNN (k=1) Accuracy: {accuracy_k1:.2f}')
print(f'kNN (k=3) Accuracy: {accuracy_k3:.2f}')

# Logistic Regression with hyperparameter tuning
log_reg_params = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
log_reg_grid = GridSearchCV(log_reg, log_reg_params, cv=5, scoring='accuracy')
log_reg_grid.fit(X_train, y_train)

# Best parameters and score for Logistic Regression
log_reg_best_params = log_reg_grid.best_params_
log_reg_best_score = log_reg_grid.best_score_

# Extract the best 'C' value
best_C = log_reg_best_params['logisticregression__C']

# Retrain and evaluate Logistic Regression on the test set
log_reg_best = make_pipeline(StandardScaler(), LogisticRegression(C=best_C, max_iter=1000))
log_reg_best.fit(X_train, y_train)
y_pred_lr = log_reg_best.predict(X_test)
accuracy_lr_test = accuracy_score(y_test, y_pred_lr)
confusion_lr = confusion_matrix(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)

# Random Forest with hyperparameter tuning
rf_params = {'n_estimators': [10, 50, 100, 200]}
rf = RandomForestClassifier()
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)

# Best parameters and score for Random Forest
rf_best_params = rf_grid.best_params_
rf_best_score = rf_grid.best_score_

# Retrain and evaluate Random Forest on the test set
rf_best = RandomForestClassifier(**rf_best_params)
rf_best.fit(X_train, y_train)
y_pred_rf = rf_best.predict(X_test)
accuracy_rf_test = accuracy_score(y_test, y_pred_rf)
confusion_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

# Print best parameters, validation and test accuracies for both models
print("Logistic Regression Best Parameters:", log_reg_best_params)
print("Logistic Regression - Validation Score:", log_reg_best_score, "Test Accuracy:", accuracy_lr_test)
print("Confusion Matrix for Logistic Regression:\n", confusion_lr)
print("Classification Report for Logistic Regression:\n", report_lr)
print("\n")
print("Random Forest Best Parameters:", rf_best_params)
print("Random Forest - Validation Score:", rf_best_score, "Test Accuracy:", accuracy_rf_test)
print("Confusion Matrix for Random Forest:\n", confusion_rf)
print("Classification Report for Random Forest:\n", report_rf)

# Analyze incorrect predictions for Random Forest
errors = np.where(y_pred_rf != y_test)[0]
print("Number of incorrect predictions by Random Forest:", len(errors))
