from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import turicreate

__author__ = 'Khashayar'
__email__ = 'khashayar@ghamati.com'


class Regression(object):

    def __init__(self, data):
        self.data = data

    def estimate_coefficients_by_ols(self, features, target):
        numinator = 0
        denuminator = 0
        for feature in features:
            vector = self.data[feature]
            target_v = self.data[target]
            numinator += sum(((vector - vector.mean()) *
                              (target_v - target_v.mean())))

            denuminator += sum(((vector - vector.mean()) ** 2))

        B2 = numinator / denuminator

        X_bar = 0
        Y_bar = 0
        for feature in features:
            vector = self.data[feature]
            target_v = self.data[target]
            X_bar += sum((vector - vector.mean()))
            Y_bar += sum(target_v - target_v.mean())

        B1 = Y_bar - B2 * X_bar

        return B1, B2

    def estimate_coefficients_by_mle(self, features, target):

        target_v = None
        X_matrix = np.eye(0, 0)
        for idx, feature in enumerate(features):
            vector = self.data[feature].to_numpy().tolist()
            target_v = self.data[target].to_numpy().tolist()

            if not bool(X_matrix.any()):
                X_matrix = np.eye(len(features) + 1, len(vector))
                X_matrix[0] = [1 for ones in range(len(vector))]

            X_matrix[idx + 1] = vector

        # MLE Estimator after derivation
        _X = X_matrix.dot(X_matrix.T)
        Y_matrix = np.array(target_v)
        result = np.linalg.inv(_X.T).dot(X_matrix.dot(Y_matrix))

        return result[0], result[1]

    def plot(self, x, y, B1, B2, title):
        plt.plot(x, y, '.', x, (B1 + B2 * x), '-')
        plt.title(title)
        plt.show()

    def print_results(self, method_name, B1, B2, rmse):
        print(f'-------{method_name}--------\n')
        print(f'----B1----B2----RMSE---\n')
        print(f'{B1} | {B2} | {rmse}\n')

    def estimate_rmse(self, Y, test_data, B1, B2, features, method):
        se = 0
        yh = 0
        for feature in features:
            vector = test_data[feature]
            yh += B2 * vector

        yh += B1

        for i in range(len(Y)):
            diff = (Y[i] - yh[i]) ** 2
            se += diff

        return sqrt(se/len(Y))


if __name__ == '__main__':
    data = turicreate.SFrame("./home_data.sframe/")
    training_set, test_set = data.random_split(.8, seed=0)

    regression = Regression(data=training_set)
    # features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']
    features = ['sqft_living',]

    erros = []

    # OLS
    B1, B2 = regression.estimate_coefficients_by_ols(features=features,
                                                     target='price')

    rmse = regression.estimate_rmse(
        test_set['price'], test_set, B1, B2, features, 'OLS')

    erros.append(('OLS', rmse))
    regression.print_results('OLS', B1, B2, rmse)

    if len(features) == 1:
        regression.plot(test_set['sqft_living'],
                        test_set['price'], B1, B2, 'OLS')


    # MLE
    B1, B2 = regression.estimate_coefficients_by_mle(
        features=features, target='price')

    rmse = regression.estimate_rmse(
        test_set['price'], test_set, B1, B2, features, 'MLE')

    erros.append(('MLE', rmse))
    regression.print_results('MLE', B1, B2, rmse)

    if len(features) == 1:
        regression.plot(
            test_set['sqft_living'], test_set['price'], B1, B2, 'MLE'
        )

    sorted_error = sorted(erros, key=lambda x: x[1])
    print("---ERRORS---\n")
    print(f"MIN : {min(sorted_error)}\n")
    print(f"MAX : {max(sorted_error)}")
