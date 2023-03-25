import numpy as np
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch

import matplotlib.pyplot as plt

def plot_dataset_on_bloch_sphere(model, X, y):
    bloch_sphere = Bloch()
    bloch_sphere.make_sphere()

    points_0 = []
    points_1 = []
    for i, point in enumerate(X):
        true_label = y[i]
        predicted_label = model.predict(point.reshape(1, -1))
        theta = np.arccos(point[0])
        phi = np.arctan2(point[1], point[0])

        if true_label == 0:
            points_0.append([theta, phi])
        else:
            points_1.append([theta, phi])

    bloch_sphere.add_points(points_0, 'b')
    bloch_sphere.add_points(points_1, 'g')

    plt.show()

def plot_decision_boundary(model, X, y, pad=1):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap = plt.cm.RdBu
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap=cmap)
    plt.show()