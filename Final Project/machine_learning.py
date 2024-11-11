"""
Project Proposal: Machine Learning Data Analysis Using Scikit-Learn and TensorFlow

Project Overview:
The "Machine Learning Data Analysis" project aims to develop a Python-based solution that leverages machine learning algorithms to analyze datasets and predict outcomes. Using libraries like Scikit-Learn and TensorFlow, the project will involve data collection, preprocessing, model development, evaluation, and visualization to derive meaningful insights and make accurate predictions.

Project Objectives:
1. **Data Collection and Preprocessing**:
   - Acquire relevant datasets from open-source platforms or specific domains (e.g., finance, healthcare, e-commerce).
   - Clean and preprocess the data to ensure quality and suitability for machine learning models.

2. **Model Development**:
   - Utilize Scikit-Learn and TensorFlow to build machine learning models for tasks such as classification, regression, or clustering.
   - Experiment with different algorithms to identify the most effective models for the dataset.

3. **Model Evaluation and Optimization**:
   - Assess model performance using appropriate metrics (e.g., accuracy, precision, recall, RMSE).
   - Optimize models through hyperparameter tuning and feature selection to enhance performance.

4. **Visualization and Reporting**:
   - Use visualization libraries like Matplotlib and Seaborn to illustrate data distributions and model results.
   - Generate comprehensive reports that interpret the findings and provide actionable insights.

5. **Deployment (Optional)**:
   - Deploy the trained models using Flask or TensorFlow Serving to create an API or web application for end-user interaction.

Scope of Work:

**Phase 1: Requirements Gathering**
- **Dataset Identification**:
  - Determine the domain of interest (e.g., healthcare outcomes, stock price prediction, customer segmentation).
  - Identify and obtain relevant datasets.
- **Define Objectives**:
  - Establish the specific machine learning tasks (e.g., predicting patient readmission, forecasting stock prices).

**Phase 2: Data Collection and Preprocessing**
- **Data Acquisition**:
  - Collect datasets from sources like Kaggle, UCI Machine Learning Repository, or industry-specific databases.
- **Data Cleaning**:
  - Handle missing values, outliers, and inconsistent data entries.
- **Data Transformation**:
  - Encode categorical variables, normalize or standardize numerical features.
  - Split data into training, validation, and test sets.

**Phase 3: Model Development**
- **Algorithm Selection**:
  - Choose suitable machine learning algorithms (e.g., linear regression, decision trees, neural networks).
- **Model Implementation**:
  - Develop models using Scikit-Learn for traditional algorithms.
  - Implement deep learning models using TensorFlow for more complex patterns.
  
**Phase 4: Model Evaluation and Optimization**
- **Performance Assessment**:
  - Evaluate models using cross-validation and appropriate performance metrics.
- **Hyperparameter Tuning**:
  - Optimize model parameters using techniques like grid search or random search.
- **Feature Engineering**:
  - Enhance model performance by creating new features or reducing dimensionality (e.g., PCA).

**Phase 5: Visualization and Reporting**
- **Data Visualization**:
  - Create plots to understand data distributions and relationships between variables.
- **Model Visualization**:
  - Illustrate model performance through learning curves, ROC curves, or confusion matrices.
- **Reporting**:
  - Compile findings into a report that explains the methodology, results, and interpretations.

**Phase 6: Deployment (Optional)**
- **Model Serialization**:
  - Save the trained models using joblib or TensorFlow SavedModel format.
- **API Development**:
  - Build a RESTful API using Flask to allow external applications to access the model.
- **Web Application**:
  - Develop a simple user interface for end-users to input data and receive predictions.

**Phase 7: Testing and Validation**
- **Functional Testing**:
  - Verify that all components (data processing, model prediction) work as intended.
- **Performance Testing**:
  - Ensure the models perform efficiently with large datasets.
- **User Acceptance Testing**:
  - Gather feedback from potential end-users and make necessary adjustments.

**Phase 8: Documentation and Handover**
- **Technical Documentation**:
  - Document the code, algorithms used, and system architecture.
- **User Guides**:
  - Create manuals on how to use the models and interpret the results.
- **Training Sessions**:
  - Conduct workshops or training sessions for stakeholders.

Deliverables:
1. **Data Processing Scripts**: Python scripts for data cleaning and preprocessing with documentation.
2. **Machine Learning Models**: Trained models saved in a reusable format.
3. **Evaluation Reports**: Detailed analysis of model performance and findings.
4. **Visualizations**: Graphs and charts illustrating data insights and model results.
5. **Deployed Application**: (Optional) A functional API or web application for model interaction.
6. **Documentation**: Comprehensive technical and user documentation.

Tools & Technologies:
- **Programming Language**: Python
- **Libraries**:
  - **Data Handling**: Pandas, NumPy
  - **Machine Learning**: Scikit-Learn, TensorFlow, Keras
  - **Visualization**: Matplotlib, Seaborn, Plotly (optional)
- **Development Environment**: Jupyter Notebooks or Integrated Development Environments (IDEs) like PyCharm
- **Deployment**: Flask, TensorFlow Serving (optional)
- **Version Control**: Git and GitHub or GitLab

Timeline:
- **Phase 1: Requirements Gathering**: 1 week
- **Phase 2: Data Collection and Preprocessing**: 2 weeks
- **Phase 3: Model Development**: 2-3 weeks
- **Phase 4: Model Evaluation and Optimization**: 2 weeks
- **Phase 5: Visualization and Reporting**: 1 week
- **Phase 6: Deployment (Optional)**: 1-2 weeks
- **Phase 7: Testing and Validation**: 1 week
- **Phase 8: Documentation and Handover**: 1 week

**Total Estimated Time**: 10-13 weeks

Risks and Mitigation:
- **Data Quality Issues**:
  - *Risk*: Poor data quality can lead to unreliable models.
  - *Mitigation*: Implement rigorous data cleaning and validation processes.

- **Overfitting Models**:
  - *Risk*: Models may perform well on training data but poorly on unseen data.
  - *Mitigation*: Use cross-validation, regularization techniques, and keep a separate test set for final evaluation.

- **Computational Resources**:
  - *Risk*: Training complex models, especially deep learning models, may require significant computational power.
  - *Mitigation*: Utilize cloud services or GPUs if necessary, and optimize code for efficiency.

- **Project Scope Creep**:
  - *Risk*: Adding features or expanding the project scope can delay timelines.
  - *Mitigation*: Stick to the defined scope and document any requested changes for future phases.

Additional Considerations:
- **Ethical AI Practices**:
  - Ensure the model does not perpetuate biases present in the data.
  - Include fairness metrics and consider ethical implications of predictions.

- **Data Privacy**:
  - Comply with data protection regulations like GDPR if using personal or sensitive data.
  - Anonymize data where necessary.

Conclusion:
This project will enhance proficiency in machine learning techniques using Scikit-Learn and TensorFlow. It offers practical experience in handling real-world data, building predictive models, and translating analytical findings into actionable insights. The final product will be a robust machine learning solution capable of providing valuable predictions and analyses in the chosen domain.
"""

## 1. **Loading Libraries and Data**
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve, auc

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # Target variable (Malignant=1, Benign=0)

# Display the first few rows of the dataset
df.head()

##2. Exploratory Data Analysis (EDA) and Visualizations

#Visualize the data to understand the distribution of the target variable and the features.

# A. Visualizing the distribution of the target variable
# Importing matplotlib
import matplotlib.pyplot as plt

# Count the number of benign and malignant cases
benign = sum(y == 0)
malignant = sum(y == 1)

# Plotting the distribution with Matplotlib
plt.figure(figsize=(6, 4))
plt.bar(['Benign', 'Malignant'], [benign, malignant], color=['blue', 'red'])

# Adding title and labels
plt.title('Distribution of Diagnosis (Malignant vs Benign)')
plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
plt.ylabel('Count')

# Display the plot
plt.show()

#B. Correlation Heatmap

# A heatmap of feature correlations helps identify multicollinearity (i.e., highly correlated features).

# Correlation heatmap to identify relationships between features
import matplotlib.pyplot as plt
import numpy as np

# Calculate the correlation matrix
corr_matrix = df.corr()

# Create a figure with a specified size
fig, ax = plt.subplots(figsize=(12, 8))

# Plotting the correlation matrix using imshow (heatmap-like effect)
cax = ax.matshow(corr_matrix, cmap='coolwarm')

# Add a colorbar for the heatmap
fig.colorbar(cax)

# Add labels to the axes
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticklabels(corr_matrix.columns)

# Add title
ax.set_title('Feature Correlation Heatmap')

# Display the plot
plt.show()

#C. Distribution of Key Features

#It's important to check the distribution of some key features like `mean radius`, `mean texture`, etc.

# Plotting histograms of key features
df[['mean radius', 'mean texture', 'mean perimeter', 'mean area']].hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()

#D. Create a figure with a specified size - purpose: Using Boxplots to detect the Outliers
plt.figure(figsize=(12, 6))

# Extract 'mean radius' values for benign (0) and malignant (1)
benign_data = df.loc[y == 0, 'mean radius']
malignant_data = df.loc[y == 1, 'mean radius']

# Create boxplot for 'mean radius' grouped by diagnosis (0: Benign, 1: Malignant)
plt.boxplot([benign_data, malignant_data], labels=['Benign (0)', 'Malignant (1)'])

# Add title and labels
plt.title('Boxplot of Mean Radius by Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Mean Radius')

# Display the plot
plt.show()

##3. Preprocessing the Data

#a. Handling Missing Data

#The dataset should not contain missing values, but we will check for completeness.

# Check for missing values
print(df.isnull().sum())

#b. Feature Scaling

#It’s good practice to 'standardize' the features, especially since we are using models like Logistic Regression or SVM.

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# View
print(X_scaled)

#c. Train-Test Split

#We will split the dataset into training and testing sets to evaluate the model’s performance.

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

##4. Feature Selection

#Let’s apply 'Recursive Feature Elimination (RFE)' using a "Logistic Regression" model to select the most important features.

#Using RFE (Recursive Feature Elimination) for feature selection
model = LogisticRegression(max_iter=10000)
selector = RFE(model, n_features_to_select=10)  # Select top 10 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get the selected features
selected_columns = df.columns[selector.support_]
print(f"Selected Features: {selected_columns}")

##5. Model Training and Evaluation

#Now we will train different models and evaluate them. We’ll start with Logistic Regression, Random Forest, and Support Vector Machine (SVM).

##a. Logistic Regression

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train_selected, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test_selected)

# Evaluate the model
print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))