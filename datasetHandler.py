from sklearn.datasets import load_iris
from KNN import *
import numpy as np 


class DatasetHandler:
    def __init__(self, data, target) -> None:
        self.data = data
        self.target = target
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def shuffle_split(self, percent):
        data_with_labels = np.column_stack((self.data, self.target))
        np.random.shuffle(data_with_labels)
        X_shuffled = data_with_labels[:, :-1]  # All columns except the last
        y_shuffled = data_with_labels[:, -1] # Last column

        ratio = int(len(self.data)*(1 - percent))
        # print(ratio)
        self.x_train = X_shuffled[:ratio]
        self.y_train = y_shuffled[:ratio].reshape(-1, 1)

        self.x_test = X_shuffled[ratio:]
        self.y_test = y_shuffled[ratio:]

    def get_train_test(self):
        return (self.x_train, self.y_train, self.x_test, self.y_test)
    
def MSE(y, y_pred):
    return np.mean((y_pred-y_test)**2)

# load the dataset:
iris = load_iris()

# handle the dataset(shuffle and split)
obj = DatasetHandler(iris.data, iris.target)
obj.shuffle_split(0.3)
x_train, y_train, x_test, y_test = obj.get_train_test()

# predict y_test
knn = KNN(3, x_train, y_train )
y_pred = [knn.prediction(point=row) for row in x_test]

# MSE:
print('MSE: ', MSE(y_test, y_pred) )







