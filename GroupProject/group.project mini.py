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
    # Use at least 2 different preprocessing techniques.

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
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('StudentPerformanceFactors.csv')

print (df)
print("First 5 rows of the dataset:\n", df.head())

print("\nSummary statistics:\n", df.describe())


df_cleaned = df.fillna({'Hours_Studied': df['Hours_Studied'].mean(), 'Exam_Score': 'Unknown'})
print("\nCleaned Data:")
print(df_cleaned)

import numpy as np
import matplotlib.pyplot as plt
df = pd.DataFrame({
    'Hours_studied': [23, 19, 25, 10, 15],
    'Exam_score': [67, 74, 68, 70, 64]
})

df.plot(kind='scatter',x='Hours_studied', y='Exam_score', title='Scatter Plot Data')
plt.show()







# Bar Plot
categories = ['Hours_Studied', 'Exam_Score']
values = [10, 20]
plt.bar(categories, values)
plt.title('Bar Plot')
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.title('Histogram')
plt.show()