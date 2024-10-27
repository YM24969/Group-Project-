# Objective: Apply data science concepts on a dataset of your choice.

# Tasks:
    # Acquire, clean, and preprocess data.
    # Perform EDA and visualize key insights.
	# Build and evaluate a machine learning model.

# Requirements:
    # Work on this as a group (Same team as the previous GIT exercise).
    # Use a dataset that is not used in the class.
    # Use at least 3 different visualization techniques.
    # Use at least 1 different machine learning algorithms.
    # Use at least 2 different evaluation metrics.
    # Use at least 2 different preprocessing techniques. - this should not be on the bottom!

# Submission Timeline:
    # Submit the code and a report in 3 weeks.
    # The report should include:
        # Introduction to the dataset.
        # Data cleaning and preprocessing steps.
        # EDA and key insights.
        # Machine learning model building and evaluation.
        # Conclusion.
        # References (if any).


import os
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import root_mean_squared_error, r2_score
import seaborn as sns


os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('StudentPerformanceFactors.csv')

print (df)

print("First 5 rows of the dataset:\n", df.head())

print("\nSummary statistics:\n", df.describe())


#Preprocessing: filling missing values with mean 
df_cleaned = df.fillna({'Hours_Studied': df['Hours_Studied'].mean(), 'Exam_Score': df['Exam_Score'].mean()})
print("\nCleaned Data:")
print(df_cleaned)

# 3. Data Preprocessing
# Convert categorical columns (Operating System, Gender) into numeric values using Label Encoding
#le = LabelEncoder()
#df['Operating System'] = le.fit_transform(df['Operating System'])
#df['Gender'] = le.fit_transform(df['Gender'])

# Define features (X) and target (y)
# Features will include all columns except 'User ID', 'Device Model', and 'User Behavior Class'
#X = df.drop(columns=['User ID', 'Device Model', 'User Behavior Class'])
#y = df['User Behavior Class']

# Split the data into training and test sets (70% train, 30% test)

x = df[['Hours_Studied']]
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#print(X_train)
#print(y_train)

# Feature scaling: Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)

#from sklearn.linear_model import LinearRegression, LogisticRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Making predictions
y_pred = lin_reg.predict(X_test_scaled)

# Evaluating the model 
r2 = r2_score(y_test, y_pred)
print(f"R^2 score: {r2}")

## Conclusion: hours studied is not a valid or strong perditor for Exam score 

#Mean Squared Error - measuring the sensitivity to errors.lower values means better preditions
rmse = root_mean_squared_error(y_test, y_pred)
print(f'root mean squared error: {rmse}')

##the goal of this evaluation is to get the RMSE result close to zero, since it is avergaing the distance of residuals. Hence, we can conclude that Hours_Studied is a poor predition of Exam_Score. 