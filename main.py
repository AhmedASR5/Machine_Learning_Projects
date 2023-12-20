import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# start of the code .


# **********************************************************************************************************************

# part 1 start:

# 1- Read the data from the csv file and split it into training set (the first 120
# examples), validation set (the next 40 examples), and testing set (the last 40
# examples). Plot the examples from the three sets in a scatter plot (each set
# encoded with a different color). Note that the plot here will be 3D plot where
# the x and y axes represent the x1 and x2 features, whereas the z-axis is the
# target label y.


# read the data
file_path = 'data_reg.csv'
data = pd.read_csv(file_path)

# split the data into train and test and validation

# train data
train_data = data.iloc[:120]

# validation data
validation_data = data.iloc[120:160]

# test data
test_data = data.iloc[160:]

# iloc is used to select the data by index number

# plot the data , to see the distribution of the data

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))  # Create a figure object
ax = fig.add_subplot(111, projection='3d')

# Plot each dataset with a different color
ax.scatter(train_data['x1'], train_data['x2'], train_data['y'], color='r',
           label='Training Set')  # * Plot the training data red
ax.scatter(validation_data['x1'], validation_data['x2'], validation_data['y'], color='g',
           label='Validation Set')  # * Plot the validation data green
ax.scatter(test_data['x1'], test_data['x2'], test_data['y'], color='b',
           label='Testing Set')  # * Plot the testing data blue

# Labeling
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('3D Scatter Plot of the Data')
ax.legend()  # Show the legend in the plot (the labels of the datasets)

# Show the plot
plt.show()

# part 1 is done
# **********************************************************************************************************************

# Apply polynomial regression on the training set with degrees in the range of 1
# to 10. Which polynomial degree is the best? Justify your answer by plotting the
# validation error vs polynomial degree curve. For each model plot the surface of
# the learned function alongside with the training examples on the same plot.
# (hint: you can use PolynomialFeatures and LinearRegression from
# scikit-learn library)

# part 2 start:


# Prepare data

X_train = train_data[['x1', 'x2']]
y_train = train_data['y']
X_val = validation_data[['x1', 'x2']]
y_val = validation_data['y']

degrees = range(1, 11)
val_errors = []

# Iterate over different polynomial degrees
for degree in degrees:
    # Generate polynomial features
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict and evaluate on the validation set
    y_val_pred = model.predict(X_val_poly)
    val_error = mean_squared_error(y_val, y_val_pred)
    val_errors.append(val_error)

# Find the best degree
best_degree = degrees[np.argmin(val_errors)]

# Plot validation error vs polynomial degree
plt.figure(figsize=(10, 6))
plt.plot(degrees, val_errors, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Validation Error')
plt.title('Validation Error vs. Polynomial Degree')
plt.show()

# Example for the best degree
degree = best_degree
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Create a mesh grid for plotting
x1_range = np.linspace(X_train['x1'].min(), X_train['x1'].max(), 100)
x2_range = np.linspace(X_train['x2'].min(), X_train['x2'].max(), 100)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
mesh_data = poly.transform(np.c_[xx1.ravel(), xx2.ravel()])
y_pred_mesh = model.predict(mesh_data).reshape(xx1.shape)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx1, xx2, y_pred_mesh, alpha=0.3)
ax.scatter(X_train['x1'], X_train['x2'], y_train, color='r')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title(f'Polynomial Regression (Degree {degree}) with Training Data')
plt.show()

# part 2 is done

# **********************************************************************************************************************

#  Apply ridge regression on the training set to fit a polynomial of degree 8. For
# the regularization parameter, choose the best value among the following
# options: {0.001, 0.005, 0.01, 0.1, 10}. Plot the MSE on the validation vs the
# regularization parameter.
# (hint: you can use Ridge regression implementation from scikit-learn)
#

# part 3 start:


# Prepare data
X_train = train_data[['x1', 'x2']]
y_train = train_data['y']
X_val = validation_data[['x1', 'x2']]
y_val = validation_data['y']

# Polynomial degree
degree = 8

# Generate polynomial features
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# Set of alpha values to try
alphas = [0.001, 0.005, 0.01, 0.1, 10]
val_errors = []

# Iterate over different alpha values
for alpha in alphas:
    # Fit a Ridge regression model
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)

    # Predict and evaluate on the validation set
    y_val_pred = model.predict(X_val_poly)
    val_error = mean_squared_error(y_val, y_val_pred)
    val_errors.append(val_error)

# Plot validation MSE vs alpha
plt.figure(figsize=(10, 6))
plt.plot(alphas, val_errors, marker='o')
plt.xscale('log')  # Since alphas vary in orders of magnitude
plt.xlabel('Alpha (Regularization parameter)')
plt.ylabel('Validation MSE')
plt.title('Validation MSE vs. Alpha for Ridge Regression')
plt.show()

# part 3 is done

# **********************************************************************************************************************

# using the logistic regression implementation of scikit-learn library, Learn
# a logistic regression model with a linear decision boundary. Draw the decision
# boundary of the learned model on a scatterplot of the training set (similar to
# Figure 1). Compute the training and testing accuracy of the learned model.


# part 4 start:


# Load the data
train_data = pd.read_csv('train_cls.csv')
test_data = pd.read_csv('test_cls.csv')

# Encode class labels
class_mapping = {'C1': 0, 'C2': 1}
train_data['class'] = train_data['class'].map(class_mapping)
test_data['class'] = test_data['class'].map(class_mapping)

# Prepare the data
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Compute accuracies
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))

# Plot the decision boundary
x_min, x_max = X_train['x1'].min() - 1, X_train['x1'].max() + 1
y_min, y_max = X_train['x2'].min() - 1, X_train['x2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train['x1'], X_train['x2'], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Print accuracies
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# part 4 is done

# **********************************************************************************************************************


# Repeat part 1 but now to learn a logistic regression model with quadratic
# decision boundary.

# part 5 start:

# Generate quadratic features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train a logistic regression model on the quadratic features
model_quad = LogisticRegression()
model_quad.fit(X_train_poly, y_train)

# Compute accuracies
train_accuracy_quad = accuracy_score(y_train, model_quad.predict(X_train_poly))
test_accuracy_quad = accuracy_score(y_test, model_quad.predict(X_test_poly))

# Plot the decision boundary for the quadratic model
x_min, x_max = X_train['x1'].min() - 1, X_train['x1'].max() + 1
y_min, y_max = X_train['x2'].min() - 1, X_train['x2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Create mesh grid for plotting
grid = np.c_[xx.ravel(), yy.ravel()]
grid_poly = poly.transform(grid)
Z_quad = model_quad.predict(grid_poly)
Z_quad = Z_quad.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_quad, alpha=0.8)
plt.scatter(X_train['x1'], X_train['x2'], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Logistic Regression with Quadratic Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Print accuracies
print(f"Training Accuracy (Quadratic): {train_accuracy_quad}")
print(f"Testing Accuracy (Quadratic): {test_accuracy_quad}")