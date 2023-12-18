import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import numpy as np

# Load the dataset
file_path = 'cars.csv'  # Replace with your file path
cars_data = pd.read_csv(file_path)

# Part 1: Number of features and examples
# we used here shape
# because it provides the dimensions of the data first element is number of rows (features)
# and the second number of columns which is the number of samples.
num_features = cars_data.shape[1]
num_examples = cars_data.shape[0]

print("Number of Features:", num_features)
print("Number of Examples:", num_examples)

# Part 2: Check for missing values
missing_values = cars_data.isnull().sum()
print(missing_values)

# Filling missing values in 'horsepower' with mean
cars_data['horsepower'].fillna(cars_data['horsepower'].mean(), inplace=True)

# Filling missing values in 'origin' with mode
cars_data['origin'].fillna(cars_data['origin'].mode()[0], inplace=True)

'''
fillna : pandas function which we used to fill missing values 
cars_data['horsepower'].mean() : getting the value of mean of all horsepower values.
inplace= True this create immediate change on the values of cars data array.
cars_data['origin'].mode() : this will give us an array with the values that appears more frequently
so we used [0] to choose the first more frequent value
Ahmed Zubaidia 1200105. 
'''

# Box plot for mpg by country (assuming 'origin' represents country)
plt.figure(figsize=(10, 6))
sns.boxplot(x='origin', y='mpg', data=cars_data)  # choosing the origin as x value and mpg as y value
plt.title('Fuel Economy (mpg) by Country of Origin')
plt.xlabel('Country of Origin')
plt.ylabel('Miles Per Gallon (mpg)')
plt.show()

plt.figure(figsize=(18, 6))

# Histogram for 'acceleration'
plt.subplot(1, 3, 1)
sns.histplot(cars_data['acceleration'], kde=True)
plt.title('Histogram of Acceleration')

# Histogram for 'horsepower'
plt.subplot(1, 3, 2)
sns.histplot(cars_data['horsepower'], kde=True)
plt.title('Histogram of Horsepower')

# Histogram for 'mpg'
plt.subplot(1, 3, 3)
sns.histplot(cars_data['mpg'], kde=True)
plt.title('Histogram of MPG')

plt.tight_layout()
plt.show()

features = ['acceleration', 'horsepower', 'mpg']

# Calculating skewness for each feature
skewness = [(feature, skew(cars_data[feature])) for feature in features]
skewness_df = pd.DataFrame(skewness, columns=['Feature', 'Skewness'])
print(skewness_df)

# Scatter plot of 'horsepower' vs 'mpg'
plt.figure(figsize=(10, 6))
plt.scatter(cars_data['horsepower'], cars_data['mpg'], alpha=0.5)
plt.title('Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (mpg)')
plt.grid(True)
plt.show()

# Calculate and print the correlation
correlation = cars_data['horsepower'].corr(cars_data['mpg'])
print("Correlation between horsepower and mpg:", correlation)

# Preparing the data for linear regression
X = np.column_stack((np.ones(len(cars_data)), cars_data['horsepower']))
y = cars_data['mpg'].values

# Closed form solution for linear regression
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# Plotting the scatter plot and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(cars_data['horsepower'], y, alpha=0.5)
plt.plot(cars_data['horsepower'], X @ theta, color='red')
plt.title('Linear Regression: Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (mpg)')
plt.grid(True)
plt.show()

# Adding the quadratic term for horsepower
X_quad = np.column_stack((np.ones(len(cars_data)), cars_data['horsepower'], cars_data['horsepower'] ** 2))
y = cars_data['mpg'].values

# Closed form solution for quadratic regression
theta_quad = np.linalg.inv(X_quad.T @ X_quad) @ X_quad.T @ y

# Generating predictions for plotting
horsepower_range = np.linspace(cars_data['horsepower'].min(), cars_data['horsepower'].max(), 100)
mpg_predictions_quad = theta_quad[0] + theta_quad[1] * horsepower_range + theta_quad[2] * horsepower_range**2

# Plotting the scatter plot and the quadratic regression line
plt.figure(figsize=(10, 6))
plt.scatter(cars_data['horsepower'], y, alpha=0.5)
plt.plot(horsepower_range, mpg_predictions_quad, color='green')
plt.title('Quadratic Regression: Horsepower vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (mpg)')
plt.grid(True)
plt.show()

# Assuming 'horsepower' needs to be scaled
# Standardize the 'horsepower' feature
horsepower_mean = cars_data['horsepower'].mean()
horsepower_std = cars_data['horsepower'].std()
cars_data['horsepower_scaled'] = (cars_data['horsepower'] - horsepower_mean) / horsepower_std

# Preparing the data for linear regression
X = np.column_stack((np.ones(len(cars_data)), cars_data['horsepower_scaled']))
y = cars_data['mpg'].values

# Gradient descent settings
alpha = 0.01  # learning rate
iterations = 1000  # number of iterations to run gradient descent
theta = np.zeros(2)  # starting values for the parameters


# Gradient descent function
def gradient_descent(X, y, theta, alpha, iterations):

    for i in range(iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = np.dot(X.T, errors)
        gradient *= (2 / len(y))
        theta -= alpha * gradient

    return theta


# Run gradient descent
theta_final = gradient_descent(X, y, theta, alpha, iterations)

# Plot the data and the linear fit
plt.scatter(cars_data['horsepower_scaled'], y, color='blue', alpha=0.5)
plt.plot(cars_data['horsepower_scaled'], X.dot(theta_final), color='red')
plt.title('Horsepower vs MPG')
plt.xlabel('Horsepower (Standardized)')
plt.ylabel('MPG')
plt.show()

print("Theta after gradient descent:", theta_final)
