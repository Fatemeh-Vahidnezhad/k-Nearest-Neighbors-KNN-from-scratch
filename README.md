# kNN Algorithm Implementation from Scratch
## Project Description
This repository contains a Python implementation of the k-Nearest Neighbors (kNN) algorithm, developed from scratch. 
It includes functionality for shuffling and splitting datasets, which is crucial for training and testing the kNN model. 
This implementation aims to provide a clear understanding of how the kNN algorithm works and can be customized to fit specific data science tasks.
The k-Nearest Neighbors (kNN) algorithm operates in several distinct steps to classify new instances based on available data:

Distance Calculation: 
First, the algorithm computes the distance between the data point in question (test data) and all the data points in the training set.
Commonly, the Euclidean distance metric is used, although other metrics like Manhattan or Hamming can also be employed depending on the application.

Sorting: Once distances are calculated, they are sorted in ascending order. This sorting process helps in identifying the nearest neighbors quickly.

Selecting Nearest Neighbors: From the sorted list of distances, the algorithm selects the top 'k' entries. These represent the 'k' closest 
training data points to the test data point.

Majority Voting: Finally, the algorithm performs a majority vote among the selected neighbors. Each point votes for their class label, 
and the class receiving the highest number of votes is chosen as the final classification for the test data point.

## Features
kNN Algorithm: Custom implementation that allows for easy adjustment of the k value.
Data Shuffling: Ensures that the data does not hold any inherent biases due to order.
Data Splitting: Separates data into training and testing sets for model evaluation.
## Prerequisites
Before you begin, ensure you have the following installed:

Python 3.6 or higher
NumPy library
You can install NumPy using pip:

pip install numpy
Installation
Clone this repository to your local machine using the following command:


git clone https://github.com/Fatemeh-Vahidnezhad/k-Nearest-Neighbors-KNN-from-scratch.git

cd knn-from-scratch
## Usage
To use this kNN implementation, you need to import the KNN class from the knn.py file (assuming your main script is named knn.py).
Here is a quick example of how to use the class:


from knn import KNN
import numpy as np
from sklearn.datasets import load_iris

#load the dataset:
iris = load_iris()

#handle the dataset(shuffle and split)
obj = DatasetHandler(iris.data, iris.target)
obj.shuffle_split(0.3)
x_train, y_train, x_test, y_test = obj.get_train_test()

#predict y_test
knn = KNN(3, x_train, y_train )
y_pred = [knn.prediction(point=row) for row in x_test]

#MSE:
print('MSE: ', MSE(y_test, y_pred) )

## Contributing
Contributions to this project are welcome! Here's how you can contribute:

Fork the repository and create your branch from main.
Make your changes and test them.
Submit a pull request detailing the changes made and their purpose.
