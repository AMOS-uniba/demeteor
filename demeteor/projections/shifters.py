import math
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike


class Shifter:
    """ Shifts without scaling or rotation """
    def __init__(self, *, x0: float = 0, y0: float = 0):
        self.x0 = x0
        self.y0 = y0

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x - self.x0, y - self.y0

    def invert(self, nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return nx + self.x0, ny + self.y0

    def as_dict(self) -> dict[str, float]:
        return dict(x=self.x0, y=self.y0)


class ScalingShifter(Shifter):
    """ Shifts and scales the sensor without rotation """
    def __init__(self, *, x0: float = 0, y0: float = 0, xs: float = 1, ys: float = 1):
        super().__init__(x0=x0, y0=y0)
        self.xs = xs    # x scaling factor
        self.ys = ys    # y scaling factor

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nx, ny = super().__call__(x, y)
        return nx * self.xs, ny * self.ys

    def invert(self, nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return super().invert(nx / self.xs, ny / self.ys)
    

class OpticalAxisShifter(Shifter):
    """ Shifts and derotates the optical axis of the sensor """
    def __init__(self, *, x0: float = 0, y0: float = 0, a0: float = 0, E: float = 0):
        super().__init__(x0=x0, y0=y0)
        self.a0 = a0                # rotation of the optical axis
        self.E = E                  # true azimuth of the centre of FoV

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = super().__call__(x, y)
        r = np.sqrt(np.square(xs) + np.square(ys))
        b = self.a0 - self.E + np.arctan2(ys, xs)
        b = np.mod(b, math.tau)
        return r, b

    def invert(self, r: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xi = b - self.a0 + self.E
        xs, ys = r * np.cos(xi), r * np.sin(xi)
        x, y = super().invert(xs, ys)
        return x, y

    def __str__(self):
        return f"<{self.__class__} x0={self.x0} y0={self.y0} a0={self.a0} E={self.E}>"

    def as_dict(self):
        return super().as_dict() | dict(
            a0=float(self.a0),
            E=float(self.E),
        )


class TiltShifter(OpticalAxisShifter):
    """
    Extends OpticalAxisShifter with imaging plane tilt.
    The optical axis is tilted with respect to the sensor normal at angle A in azimuth F.

    For further details see
    Borovička (1995): A new positional astrometric method for all-sky cameras.
    This method has been optimized for sanity
    """
    def __init__(self, *, x0: float = 0, y0: float = 0, a0: float = 0, A: float = 0, F: float = 0, E: float = 0):
        super().__init__(x0=x0, y0=y0, a0=a0, E=E)
        assert -1 <= A <= 1, f"Invalid parameter {A=}: must be -1 <= A <= 1."

        self.A = A                  # imaging plane tilt, amplitude
        self.F = F                  # imaging plane tilt, phase
        self._phi = self.F - self.a0
        self._cos_term = np.cos(self._phi)
        self._sin_term = np.sin(self._phi)

    def __call__(self,
                 x: np.ndarray,
                 y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = x - self.x0
        ys = y - self.y0
        r, b = super().__call__(x, y)
        r += self.A * (ys * self._cos_term - xs * self._sin_term)
        return r, b

    def invert(self,
               r: ArrayLike,
               b: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        xi = b - self.a0 + self.E
        denom = 1 + self.A * np.sin(xi - self._phi)
        x = self.x0 + r * np.cos(xi) / denom
        y = self.y0 + r * np.sin(xi) / denom
        return x, y

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} x0={self.x0} y0={self.y0} a0={self.a0} A={self.A} F={self.F} E={self.E}>"

    def as_dict(self):
        return super().as_dict() | dict(
            A=float(self.A),
            F=float(self.F),
        )
