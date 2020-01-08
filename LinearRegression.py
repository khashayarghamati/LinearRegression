__author__ = 'Khashayar'
__email__ = 'khashayar@ghamati.com'

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):

    def __init__(self, features):
        self.features = features

    def estimate_coefficients(self):
        X = []
        Y = []
        ones = []

        for feature in self.features:
            ones.append(1)

            x = feature[0]
            y = feature[1]

            X.append(x)
            Y.append(y)

        X_matrix = np.eye(2, len(X))
        X_matrix[0] = ones
        X_matrix[1] = X

        _X = X_matrix.dot(X_matrix.T)
        Y_matrix = np.array(Y)
        result = np.linalg.inv(_X.T).dot(X_matrix.dot(Y_matrix))

        regress = np.multiply(result[1], X) + result[0]
        plt.plot(X, Y, 'ro', X, regress)
        plt.show()

        print (result)


if __name__ == '__main__':

    data = [
        (1.47, 52.21),
        (1.50, 53.12),
        (1.52, 54.48),
        (1.55, 55.84),
        (1.57, 57.20),
        (1.60, 58.57),
        (1.63, 59.93),
        (1.65, 61.29),
        (1.68, 63.11),
        (1.70, 64.47),
        (1.73, 66.28),
        (1.75, 68.10),
        (1.78, 69.92),
        (1.80, 72.19),
        (1.83, 74.46),
    ]

    lr = LinearRegression(features=data)
    lr.estimate_coefficients()

    data1 = [

        (2, 258.72),
        (8, 345.15),
        (18, 468.57),
        (26, 932),
        (28, 795.97),
        (32, 1099.8),
        (51, 2720.2),
        (84, 7173.6),
        (94, 8951.2),
        (97, 9495.2),
    ]

    lr1 = LinearRegression(features=data1)
    lr1.estimate_coefficients()

