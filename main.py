import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# start of the code .


#**********************************************************************************************************************

#part 1 start:

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
fig = plt.figure(figsize=(10, 8)) # Create a figure object
ax = fig.add_subplot(111, projection='3d') # * Create an axes object in the figure with 1 row and 1 column and the index 1 (111) ,and make it 3D (projection='3d')


# Plot each dataset with a different color
ax.scatter(train_data['x1'], train_data['x2'], train_data['y'], color='r', label='Training Set') # * Plot the training data red
ax.scatter(validation_data['x1'], validation_data['x2'], validation_data['y'], color='g', label='Validation Set') # * Plot the validation data green
ax.scatter(test_data['x1'], test_data['x2'], test_data['y'], color='b', label='Testing Set') # * Plot the testing data blue

# Labeling
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('3D Scatter Plot of the Data')
ax.legend() # Show the legend in the plot (the labels of the datasets)

# Show the plot
plt.show()


# part 1 is done
#**********************************************************************************************************************

