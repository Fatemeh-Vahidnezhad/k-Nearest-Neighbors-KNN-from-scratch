import numpy as np 


class KNN:
    def __init__(self, k, dataset_train, target) -> None:
        self.k = k
        self.train = dataset_train
        self.target = target
        self.distance = None

    def euclidean_distance(self, point):
        return np.sqrt( np.sum((self.train-point)**2, axis = 1))
         

    def sort_distance(self, distances):
        indices = np.argsort(distances)
        return np.hstack([distances[indices][:, np.newaxis], self.target[indices]])[:self.k]

    def common_label(self, data):
        label, counts = np.unique(data[:, 1], return_counts=True)
        return label[np.argmax(counts)]

    def prediction(self, point):
        distances = self.euclidean_distance(point) 
        data = self.sort_distance(distances)
        return self.common_label(data)


# point = np.array([1,2,3])
# arr = np.array([[1,2,3],
#                 [4,5,6], 
#                 [7,8,9],
#                 [10,11,12]])
# target = np.array([[1],
#                    [0],
#                    [0],
#                    [0]])
# knn = KNN(k=3, dataset_train=arr, target=target)
# # knn.euclidean_distance(point) 
# # data = knn.sort_distance()  
# print(knn.prediction(point=point))

