import pytest

import numpy as np

from demeteor.trajectory import RayMinimizer


@pytest.fixture
def cube():
    return np.array(
        [
            [[0, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0]],
            [[0, 1, 0], [0, 0, 1]],
            [[0, 1, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0]],
            [[1, 1, 1], [0, 0, 1]],
            [[1, 1, 1], [0, 1, 0]],
            [[1, 1, 1], [1, 0, 0]],
        ]
    )


@pytest.fixture
def square():
    return np.array(
        [
            [[0, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 1, 1], [0, 0, 1]],
            [[0, 1, 1], [0, 1, 0]],
        ]
    )

@pytest.fixture
def cube_minimizer(cube):
    return RayMinimizer(cube[:, 0, :], cube[:, 1, :])


@pytest.fixture
def square_minimizer(square):
    return RayMinimizer(square[:, 0, :], square[:, 1, :])


class TestSquare:
    def test_vertex_lin(self, square, square_minimizer):
        assert square_minimizer.sum_distance(np.array([0, 0, 0])) == pytest.approx(2, rel=1e-12)

    def test_vertex_quad(self, square, square_minimizer):
        assert square_minimizer.sum_quad_distance(np.array([0, 0, 0])) == pytest.approx(2, rel=1e-12)

    def test_centre_lin(self, square, square_minimizer):
        assert square_minimizer.sum_distance(np.array([0, 0.5, 0.5])) == pytest.approx(2, rel=1e-12)

    def test_centre_quad(self, square, square_minimizer):
        assert square_minimizer.sum_quad_distance(np.array([0, 0.5, 0.5])) == pytest.approx(1, rel=1e-12)

    def test_minimize(self, square):
        assert np.allclose(RayMinimizer(square[:, 0, :], square[:, 1, :]).nearest(), np.array([0.0, 0.5, 0.5]))


class TestCube:
    def test_vertex_lin(self, cube, cube_minimizer):
        assert cube_minimizer.sum_distance(np.array([0, 0, 0])) == pytest.approx(6 + 3 * np.sqrt(2), rel=1e-12)

    def test_vertex_quad(self, cube, cube_minimizer):
        assert cube_minimizer.sum_quad_distance(np.array([0, 0, 0])) == pytest.approx(12, rel=1e-12)

    def test_centre_lin(self, cube, cube_minimizer):
        assert cube_minimizer.sum_distance(np.array([0.5, 0.5, 0.5])) == pytest.approx(6 * np.sqrt(2), rel=1e-12)

    def test_centre_quad(self, cube, cube_minimizer):
        assert cube_minimizer.sum_quad_distance(np.array([0.5, 0.5, 0.5])) == pytest.approx(6, rel=1e-12)

    def test_minimize(self, cube):
        assert np.allclose(RayMinimizer(cube[:, 0, :], cube[:, 1, :]).nearest(), np.array([0.5, 0.5, 0.5]))


class TestZero:
    def test_triline(self):
        # Three lines passing through [1, 2, 3] -> 0
        rays = RayMinimizer(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), np.array([[4, 3, 1], [7, 1, 0], [-2, -3, 4]]))
        assert np.allclose(rays.nearest(), np.array([1, 2, 3]))

    def test_quadline(self):
        # Four lines passing through [-1, 0, 3] -> 0
        rays = RayMinimizer(
            np.array([[-1, 7, 3], [-1, 0, 3], [-1, 4, 11], [2, 3, 6]]),
            np.array([[0, 1, 0], [6, -1, -3], [0, 1, 2], [1, 1, 1]]),
        )
        assert np.allclose(rays.nearest(), np.array([-1, 0, 3]))


class TestMisc:
    def test_equilateral(self):
        # Equilateral triangle centered on origin
        a = np.array([np.cos(1.2 / 3 * np.pi), np.sin(1.2 / 3 * np.pi), -5])
        b = np.array([np.cos(3.2 / 3 * np.pi), np.sin(3.2 / 3 * np.pi), -5])
        c = np.array([np.cos(5.2 / 3 * np.pi), np.sin(5.2 / 3 * np.pi), -5])
        rays = RayMinimizer.from_points(
            np.array([a, b, c]),
            np.array([b, c, a]),
        )
        assert np.allclose(rays.nearest(), np.array([0, 0, -5]), atol=1e-6)
