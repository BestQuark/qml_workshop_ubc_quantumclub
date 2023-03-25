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
