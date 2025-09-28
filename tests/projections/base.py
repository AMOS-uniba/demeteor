import numpy as np

import pytest


def pytest_generate_tests(metafunc):
    if hasattr(metafunc.cls, 'params'):
        funcarglist = metafunc.cls.params.get(metafunc.function.__name__, None)

        if funcarglist:
            argnames = sorted(funcarglist[0])
            metafunc.parametrize(
                argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
            )


class BaseTestProjection:
    projection = None

    grid = [
        dict(x=x, y=y)
        for x in np.linspace(-1, 1, 25)
        for y in np.linspace(-1, 1, 25)
        if x**2 + y**2 <= 1
    ]

    params = dict(
        test_inversion=grid,
    )

    @staticmethod
    def compare_inverted(projection, x, y, atol=1e-9):
        """ Test whether p^(-1) p (x, y) == x, y
        """
        computed = projection.invert(*projection(x, y))
        assert computed == pytest.approx((x, y), abs=atol), \
            f"Computed {computed}, expected {x, y}"

    def test_inversion(self, x, y, atol=1e-9):
        self.compare_inverted(self.projection, x, y, atol)