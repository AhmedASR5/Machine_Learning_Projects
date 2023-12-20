import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Ahmed Zubaidia 1200105

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

data = pd.read_csv('data_reg.csv')

# split the data into train and test and validation

train_data = data.iloc[:120] # train data from 0 to 120

validation_data = data.iloc[120:160] # validation data from 120 to 160

test_data = data.iloc[160:] # test data from 160 to the end



# ploting the data in 3D

fig = plt.figure(figsize=(10, 8))  # Create a figure object with size 10x8 inches.
ax = fig.add_subplot(111, projection='3d')

# Plot each dataset with a different color
ax.scatter(train_data['x1'], train_data['x2'], train_data['y'], color='r', label='Training Set')  # * Plot the training data red

ax.scatter(validation_data['x1'], validation_data['x2'], validation_data['y'], color='g', label='Validation Set')  # * Plot the validation data green

ax.scatter(test_data['x1'], test_data['x2'], test_data['y'], color='b', label='Testing Set')  # * Plot the testing data blue

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

X_train = train_data[['x1', 'x2']] # here i will get the first two columns of the data which are the features
y_train = train_data['y'] # here i will get the last column of the data which is the actual value of y
X_val = validation_data[['x1', 'x2']] # same here
y_val = validation_data['y']  # same here


validation_error_list = [] # list to save the validation error of each degree to plot it later.

# for loop to iterate over the degrees from 1 to 10
for degree in range(1, 11):

    feature_degree_function = PolynomialFeatures(degree) # here make functions according to the degree, to get the target feature when we substitute the data in it.

    X_train_polynomial = feature_degree_function.fit_transform(X_train) # here i will substitute the train data in the function to get the target feature

    X_val_polynomial = feature_degree_function.transform(X_val) # same here but with validation data


    model = LinearRegression() #  makeing the model of linear regression to substitute the data in it to get the target feature.

    model.fit(X_train_polynomial, y_train) # here i will substitute the data in the model to train it.


    y_val_pred = model.predict(X_val_polynomial) # here will substitute the validation data in the model to predict the target feature.

    validation_error = mean_squared_error(y_val, y_val_pred) # now i will compute the mean squared error of the validation data.

    validation_error_list.append(validation_error)  # adding the error to the list to plot it later.

    # determine the range of x1 and x2 to plot the surface
    x1_range = np.linspace(X_train['x1'].min(), X_train['x1'].max(), 100)
    x2_range = np.linspace(X_train['x2'].min(), X_train['x2'].max(), 100)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    mesh_data = feature_degree_function.transform(np.c_[xx1.ravel(), xx2.ravel()])
    y_pred_mesh = model.predict(mesh_data).reshape(xx1.shape)

    # plotting  the surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx1, xx2, y_pred_mesh, alpha=0.3)
    ax.scatter(X_train['x1'], X_train['x2'], y_train, color='r')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title(f'Polynomial Regression (Degree {degree}) with Training Data')
    plt.show()

# Plot validation error vs polynomial degree
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), validation_error_list, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Validation Error')
plt.title('Validation Error vs. Polynomial Degree')
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


# loading the data from the csv file. 1200105

train_data = pd.read_csv('train_cls.csv')  # train data that we will use it to make logistic model
test_data = pd.read_csv('test_cls.csv')  # data that we will use to test the model

mapped_values = {'C1': 0, 'C2': 1}  # here I used hash map to map the classes  of C1 to 0 and C2 to 1
# in order to make it easier to work with the data.

train_data['class'] = train_data['class'].map(mapped_values) # here I changed the classes according to the hash map

test_data['class'] = test_data['class'].map(mapped_values)# here also changed classes to 0 and 1 in test data

# Prepare the data
X_train = train_data.drop('class', axis=1) # here i sliced the data to get just the first two columns of features
y_train = train_data['class']  # here i saved the last column of the data which is the class
X_test = test_data.drop('class', axis=1) # same here
y_test = test_data['class'] # same here

# start making the model of logistic regression
model = LogisticRegression() # creating the model of logistic regression ,  1/ 1+ e^-(B0+B1x1.....)

model.fit(X_train, y_train) # here I substituted the data to the model to train it  the two features , and classes .



# here i will plot the decision boundary of the model

x_min = X_train['x1'].min() - 1 # here i will get the minimum value of x1 and subtract 1 from it to ensure that the plot covers all the data
x_max = X_train['x1'].max() + 1  # here i will get the maximum value of x1 and add 1 to it to ensure that the plot covers all the data
y_min = X_train['x2'].min() - 1 # same
y_max = X_train['x2'].max() + 1 # same

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)) # here i will create a mesh grid to plot the data

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # here i will predict the data using the model that i created

Z = Z.reshape(xx.shape) # here i will reshape the data to fit the plot  , to make it 2D.

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train['x1'], X_train['x2'], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# finding  accuracies
train_accuracy = accuracy_score(y_train, model.predict(X_train)) # here i will compute the accuracy of the model on the train data to print it to cmpare it with the test data
test_accuracy = accuracy_score(y_test, model.predict(X_test)) # here i will compute the accuracy of the model on the test data.

# printing  accuracies
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# part 4 is done

# **********************************************************************************************************************


# Repeat part 1 but now to learn a logistic regression model with quadratic
# decision boundary.

# part 5 start:

# Generate quadratic features
features_degree = PolynomialFeatures(degree=2, include_bias=False) # here i make the funcions  features quadratic x1^2 , x2^2 , x1x2 , x1 , x2 , 1

X_train_features_degree = features_degree.fit_transform(X_train) # i substituted the train data in function to get the quadratic features
X_test_features_degree = features_degree.transform(X_test) # same here but with test data

# make the model of logistic regression

model_quad = LogisticRegression() # same as before
model_quad.fit(X_train_features_degree, y_train) # same as before but with the quadratic features.


x_min = X_train['x1'].min() - 1 # here i will get the minimum value of x1 and subtract 1 from it to ensure that the plot covers all the data
x_max = X_train['x1'].max() + 1  # here i will get the maximum value of x1 and add 1 to it to ensure that the plot covers all the data
y_min = X_train['x2'].min() - 1 # same
y_max = X_train['x2'].max() + 1 # same

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Create mesh grid for plotting
grid = np.c_[xx.ravel(), yy.ravel()]
grid_poly = features_degree.transform(grid)
Z_quad = model_quad.predict(grid_poly)
Z_quad = Z_quad.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_quad, alpha=0.8)
plt.scatter(X_train['x1'], X_train['x2'], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Logistic Regression with Quadratic Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# finding accuracies
train_accuracy_quad = accuracy_score(y_train, model_quad.predict(X_train_features_degree))
test_accuracy_quad = accuracy_score(y_test, model_quad.predict(X_test_features_degree))

# Print accuracies
print(f"training accuracy of quadratic: {train_accuracy_quad}")
print(f"testing accuracy of quadratic: {test_accuracy_quad}")
