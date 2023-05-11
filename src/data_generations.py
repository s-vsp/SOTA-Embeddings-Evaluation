import numpy as np
import matplotlib.pyplot as plt

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

    def make_hypersphere(self) -> np.ndarray:
        """
        Method to generate a toy dataset of N-dimensional sphere.

        Returns:
            - points: points distributed on the hypersphere manifold
        """
        points = np.random.normal(loc=0., scale=1., size=(self.n_points, self.n_dims))

        r = np.sqrt(np.sum(points**2, axis=1))
        points = points / r.reshape(-1,1)

        return points

    def make_hyperball(self, radius: float=1.) -> np.ndarray:
        """
        Method to generate a toy dataset of N-dimensional ball.

        Args:
            - radius: radius of the hyperball

        Returns:
            - points: points distributed on the hyperball manifold
        """
        points = np.random.uniform(low=-1., high=1., size=(self.n_points, self.n_dims))

        r = np.sqrt(np.sum(points**2, axis=1))
        points = points / r.reshape(-1,1)
        points = points * radius

        u = np.random.uniform(low=0., high=1., size=(self.n_points)) ** (1 / self.n_dims)

        points = points * u.reshape(-1,1) * radius
    
        return points

    def make_hypertorus(self) -> np.ndarray:
        """
        Method to generate a toy dataset of N-dimensional torus.

        Returns:
            - points: points distributed on the hypertorus manifold
        """
        pass


def visualize2D(data: np.ndarray):
    # TODO
    plt.figure(figsize=(7,7))
    plt.scatter(data[:, 0], data[:, 1], color="navy")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def visualize3D(data: np.ndarray):
    # TODO
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:,0], data[:,1], data[:,2], s=5, c="navy")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

if __name__ == "__main__":
    generator = SyntheticDataGenerator(n_dims=3, n_points=10000)
    sphere = generator.make_hypersphere()
    ball = generator.make_hyperball()

    visualize3D(ball)
    visualize3D(sphere)
