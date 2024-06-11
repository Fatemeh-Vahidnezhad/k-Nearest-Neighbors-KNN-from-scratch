# kNN Algorithm Implementation from Scratch
## Project Description
This repository contains a Python implementation of the k-Nearest Neighbors (kNN) algorithm, developed from scratch. It includes functionality for shuffling and splitting datasets, which is crucial for training and testing the kNN model. This implementation aims to provide a clear understanding of how the kNN algorithm works and can be customized to fit specific data science tasks.

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

#Example Data (replace with your data)
X_train = np.array([[1, 2], [3, 4], [5, 6]])
y_train = np.array([0, 1, 0])
X_test = np.array([[2, 3]])

#Create and train the kNN classifier
classifier = KNN(k=3)
classifier.fit(X_train, y_train)

#Predict the class of the test instance
prediction = classifier.predict(X_test)
print("Predicted class:", prediction)
## Contributing
Contributions to this project are welcome! Here's how you can contribute:

Fork the repository and create your branch from main.
Make your changes and test them.
Submit a pull request detailing the changes made and their purpose.
