import math
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import numpy.typing as npt


class Shifter(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def invert(self, nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass


class ScalingShifter(Shifter):
    """ Shifts and scales the sensor without rotation """
    def __init__(self, *, x0: float = 0, y0: float = 0, xs: float = 1, ys: float = 1):
        self.x0 = x0
        self.y0 = y0
        self.xs = xs
        self.ys = ys

    def __call__(self, x: np.ndarray, y: np.ndarray):
        xs = (x - self.x0) * self.xs
        ys = (y - self.y0) * self.ys
        return xs, ys

    def invert(self, nx: np.ndarray, ny: np.ndarray):
        x = self.x0 + nx / self.xs
        y = self.y0 + ny / self.ys
        return x, y


class OpticalAxisShifter(Shifter):
    """ Shifts and derotates the optical axis of the sensor """
    def __init__(self, *, x0: float = 0, y0: float = 0, a0: float = 0, E: float = 0):
        self.x0 = x0                # x position of the centre of the optical axis
        self.y0 = y0                # y position of the centre of the optical axis
        self.a0 = a0                # rotation of the optical axis
        self.E = E                  # true azimuth of the centre of FoV

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = x - self.x0
        ys = y - self.y0
        r = np.sqrt(np.square(xs) + np.square(ys))
        b = self.a0 - self.E + np.arctan2(ys, xs)
        b = np.mod(b, math.tau)
        return r, b

    def invert(self, r: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xi = b - self.a0 + self.E
        x = self.x0 + r * np.cos(xi)
        y = self.y0 + r * np.sin(xi)
        return x, y

    def __str__(self):
        return f"<{self.__class__} x0={self.x0} y0={self.y0} a0={self.a0} E={self.E}>"

    def as_dict(self):
        return dict(
            x0=float(self.x0),
            y0=float(self.y0),
            a0=float(self.a0),
            E=float(self.E),
        )


class TiltShifter(OpticalAxisShifter):
    """
        Extends OpticalAxisShifter with imaging plane tilt.
        The optical axis is tilted with respect to the sensor normal at angle A in azimuth F.
        For further details see Borovička (1995): A new positional astrometric method for all-sky cameras.
    """
    def __init__(self, *, x0: float = 0, y0: float = 0, a0: float = 0, A: float = 0, F: float = 0, E: float = 0):
        super().__init__(x0=x0, y0=y0, a0=a0, E=E)
        self.A = A                  # imaging plane tilt, amplitude
        self.F = F                  # imaging plane tilt, phase
        self.cos_term = np.cos(self.F - self.a0)
        self.sin_term = np.sin(self.F - self.a0)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r, b = super().__call__(x, y)
        r += self.A * ((y - self.y0) * self.cos_term - (x - self.x0) * self.sin_term)
        return r, b

    def _jacobian(self, vec, r, b) -> npt.NDArray:
        """ Jacobian for the numerical inversion method """
        xs = vec[0] - self.x0
        ys = vec[1] - self.y0
        r2 = np.square(xs) + np.square(ys)
        drdx = xs / np.sqrt(r2) - self.A * self.sin_term
        drdy = ys / np.sqrt(r2) + self.A * self.cos_term
        dbdx = -ys / r2
        dbdy = xs / r2
        return np.array([
            [drdx, drdy],
            [dbdx, dbdy],
        ])

    def _func(self, vec, r: np.ndarray[float], b: np.ndarray[float]) -> (np.ndarray[float], np.ndarray[float]):
        q = self.__call__(vec[0], vec[1])
        return q[0] - r, np.mod(q[1] - b + math.pi, math.tau) - math.pi

    def invert(self, r: np.ndarray[float], b: np.ndarray[float]) -> (np.ndarray[float], np.ndarray[float]):
        vec = sp.optimize.root(
            self._func,
            np.stack(super().invert(r, b)),
            method='lm',
            args=(r, b),
            jac=self._jacobian,
            tol=1e-9,
        )
        x = vec.x[0]
        y = vec.x[1]
        return x, y

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} x0={self.x0} y0={self.y0} a0={self.a0} A={self.A} F={self.F} E={self.E}>"

    def as_dict(self):
        return super().as_dict() | dict(
            A=float(self.A),
            F=float(self.F),
        )
