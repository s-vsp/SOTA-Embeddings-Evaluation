import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

from typing import Tuple


class SyntheticDataGenerator:
    """
    Class object to generate synthetic toy datasets of n-dimensional manifolds.

    Args:
        - n_dims: dimensionality of the Euclidean space
        - n_points: number of points uniformly distributed in the manifold
    """

    def __init__(self, n_dims: int, n_points: int) -> None:
        self.n_dims = n_dims
        self.n_points = n_points

    def make_hypersphere(self, radius: float = 1.) -> np.ndarray:
        """
        Method to generate a toy dataset of N-dimensional sphere.

        Args:
            - radius: radius of the hypersphere

        Returns:
            - points: points distributed on the hypersphere manifold
        """
        points = np.random.normal(loc=0., scale=1., size=(self.n_points, self.n_dims))

        r = np.sqrt(np.sum(points ** 2, axis=1))
        points = points / r.reshape(-1, 1)

        return points * radius

    def make_hyperball(self, radius: float = 1.) -> np.ndarray:
        """
        Method to generate a toy dataset of N-dimensional ball.

        Args:
            - radius: radius of the hyperball

        Returns:
            - points: points distributed on the hyperball manifold
        """
        points = np.random.uniform(low=-1., high=1., size=(self.n_points, self.n_dims))

        r = np.sqrt(np.sum(points ** 2, axis=1))
        points = points / r.reshape(-1, 1)
        points = points * radius

        u = np.random.uniform(low=0., high=1., size=(self.n_points)) ** (1 / self.n_dims)  # noqa

        points = points * u.reshape(-1, 1) * radius

        return points

    def make_hypertorus(self, r: float = 1., R: float = 2.) -> np.ndarray:  # noqa
        """
        Method to generate a toy dataset of N-dimensional torus.

        Args:
        - r: radius of the cross-section circle of the torus
        - R: distance from the center of the torus to the center of the cross-section circle

        Returns:
            - points: points distributed on the hypertorus manifold
        """
        points = self.make_hypersphere(radius=r)
        points[:, :-1] += R

        if self.n_dims > 1:
            rotation_matrix = np.identity(self.n_dims)
            thetas = np.random.uniform(0, 2 * np.pi, self.n_points)

            for idx, theta in enumerate(thetas):
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                rotation_matrix[0, 0] = cos_theta
                rotation_matrix[0, 1] = -1 * sin_theta
                rotation_matrix[1, 0] = sin_theta
                rotation_matrix[1, 1] = cos_theta

                points[idx, :] = points[idx, :] @ rotation_matrix

        return points

    def make_triple_hypersphere_dataset(self, save: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to create a dataset of synthetic hyperspheres.

        Returns:
            - X: (n_points, n_dims)-dimensional data of hyperspheres
            - y: (n_points,)-dimensional labels corresponding to inner hyperspheres
        """
        hypersphere1 = self.make_hypersphere() * 0.5
        hypersphere2 = self.make_hypersphere()
        hypersphere3 = self.make_hypersphere() * 2.0

        labels1 = np.zeros(shape=hypersphere1.shape[0])
        labels2 = np.ones(shape=hypersphere2.shape[0])
        labels3 = np.ones(shape=hypersphere3.shape[0]) * 2

        hyperspheres = np.concatenate([hypersphere1, hypersphere2, hypersphere3], axis=0)
        labels = np.concatenate([labels1, labels2, labels3], axis=0)

        data = np.column_stack((hyperspheres, labels))
        np.random.shuffle(data)

        X, y = data[:, :-1], data[:, -1]
        if save:
            np.save(f"./data/triple_hyperspheres-n_points-{X.shape[0]}-n_dims-{X.shape[1]}", data)

        return X, y

    def make_triple_hyperball_dataset(self, save: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to create a dataset of synthetic hyperballs.

        Returns:
            - X: (n_points, n_dims)-dimensional data of hyperballs
            - y: (n_points,)-dimensional labels corresponding to inner hyperballs
        """
        hyperball1 = self.make_hyperball() * 0.5
        hyperball2 = self.make_hyperball()
        hyperball3 = self.make_hyperball() * 2.0

        labels1 = np.zeros(shape=hyperball1.shape[0])
        labels2 = np.ones(shape=hyperball2.shape[0])
        labels3 = np.ones(shape=hyperball3.shape[0]) * 2

        hyperballs = np.concatenate([hyperball1, hyperball2, hyperball3], axis=0)
        labels = np.concatenate([labels1, labels2, labels3], axis=0)

        data = np.column_stack((hyperballs, labels))
        np.random.shuffle(data)

        X, y = data[:, :-1], data[:, -1]
        if save:
            np.save(f"./data/triple_hyperballs-n_points-{X.shape[0]}-n_dims-{X.shape[1]}", data)

        return X, y


def visualize2D(data: np.ndarray):
    # TODO
    plt.figure(figsize=(7, 7))
    plt.scatter(data[:, 0], data[:, 1], color="navy")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def visualize3D(data: np.ndarray, lim3d: float = 0):
    # TODO
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5, c="navy")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if lim3d != 0:
        lim3d *= 2
        ax.set_xlim3d(-lim3d, lim3d)
        ax.set_ylim3d(-lim3d, lim3d)
        ax.set_zlim3d(-lim3d, lim3d)
    plt.show()


if __name__ == "__main__":
    N_DIMS = list(range(3,21))
    N_POINTS = [100, 500, 1000, 2500, 5000, 10_000, 20_000, 30_000]

    for n_dims in N_DIMS:
        for n_points in N_POINTS:
            data_generator = SyntheticDataGenerator(n_dims=n_dims, n_points=n_points)
            X, y = data_generator.make_triple_hypersphere_dataset(True)