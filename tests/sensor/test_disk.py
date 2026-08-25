"""
The disk, and the one property it has to have: going onto it and back is the identity.

The quarter turn and the sign on y are a display convention that leaked into the arithmetic long
ago. It is harmless only because both directions agree about it, and that agreement is what this
checks -- it is the kind of thing that survives a move to a new home and then does not.
"""
import numpy as np

from demeteor.sensor import altaz_to_disk, disk_to_altaz
from ..base import pytest_generate_tests



class TestAngularFunction:
    params = dict(
        test_inverse=[
            dict(x=x, y=y)
            for x in np.linspace(-1, 1, 5)
            for y in np.linspace(-1, 1, 7)
            if x**2 + y**2 < 1
        ],
    )

    def test_inverse(self, x, y):
        point = np.array([[x, y]])
        assert np.allclose(altaz_to_disk(disk_to_altaz(point)), point)

