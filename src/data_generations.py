import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

        Torus parameterization:
            - https://math.stackexchange.com/questions/358825/parametrisation-of-the-surface-a-torus
        """
        theta = np.random.uniform(0, 2 * np.pi, self.n_points)
        phi = np.random.uniform(0, 2 * np.pi, self.n_points)

        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)

        points = np.stack((x, y, z), axis=-1)

        return points


def visualize2D(data: np.ndarray):
    # TODO
    plt.figure(figsize=(7, 7))
    plt.scatter(data[:, 0], data[:, 1], color="navy")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def visualize3D(data: np.ndarray):
    # TODO
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5, c="navy")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    generator = SyntheticDataGenerator(n_dims=3, n_points=10_000)
    sphere = generator.make_hypersphere()
    ball = generator.make_hyperball()
    torus = generator.make_hypertorus()

    visualize3D(ball)
    visualize3D(sphere)
    visualize3D(torus)
