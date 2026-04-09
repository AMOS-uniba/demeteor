import numpy as np
import scipy as sp


class RayMinimizer:
    """
    Class that evaluates the distance of a point from a set of lines, and finds the point
    from which squared distance to all the lines is smallest.

    Has two constructors: __init__, which expectes two ndarrays of same shape, points and vectors defining the line,
    and a static `from_points` that expects two arrays of points.
    """
    def __init__(self, points: np.ndarray, vectors: np.ndarray, *, normalize: bool = False):
        assert points.shape == vectors.shape, "Points and vectors must have the same shape"

        self.points = points
        self.vectors = vectors

        if normalize:
            self.vectors /= np.linalg.norm(self.vectors)

        assert np.min(np.abs(np.linalg.norm(self.vectors)) != 0), "All vectors must be non-zero!"

    @staticmethod
    def from_points(points: np.ndarray, points2: np.ndarray):
        return RayMinimizer(points, points2 - points, normalize=False)

    def sum_distance(self, point: np.array) -> float:
        p = point - self.points
        dist = np.linalg.norm(np.cross(p, self.vectors), axis=1, ord=2) / np.linalg.norm(self.vectors, axis=1, ord=2)
        return np.sum(dist)

    def sum_quad_distance(self, point: np.ndarray) -> float:
        p = point - self.points
        dist = np.linalg.norm(np.cross(p, self.vectors), axis=1, ord=2) / np.linalg.norm(self.vectors, axis=1, ord=2)
        return np.sum(np.square(dist))

    def nearest(self) -> np.ndarray:
        return sp.optimize.minimize(self.sum_quad_distance, np.array([0, 0, 0]).T, method="L-BFGS-B").x