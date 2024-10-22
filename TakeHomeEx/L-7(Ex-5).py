import numpy as np # type: ignore
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Set random seed for reproducibility
np.random.seed(42)

# Define dataset size
# data_size = ?
data_size = 100
# Exercise 5: Generate a sample dataset and display a pairplot
# Create a sample dataset with numpy np.random.randn(), np.random.rand(), np.random.randint()
data = pd.DataFrame({
    'Variable_A': np.random.randn(data_size) ,
    'Variable_B': np.random.rand(data_size) ,
    'Variable_C': np.random.randint(1, 100, data_size),
    'Variable_D': np.random.randn(data_size) * 50 + 100,
})

# Create a pairplot of the dataset
sns.pairplot(data)
# Show the plot
plt.show()
# Optionally, save the plot as a PNG file
# plt.savefig('pairplot_output.png')