import numpy as np
import dotmap
import yaml
from typing import Union
from numpy.typing import NDArray

from .base import normalise_angles, Projection
from .shifters import TiltShifter
from .transformers import SaneBiexponentialTransformer
from .zenith import ZenithShifter


class KoniferkaProjection(Projection):
    """
    Improved Borovička all-sky projection, with correct dimensions.
    Namely,
        r1 = 1 / D,
        r2 = sqrt(1 / Q).
    """
    bounds = np.array((
        (None, None),  # x0
        (None, None),  # y0
        (None, None),  # a0
        (None, None),  # A
        (None, None),  # F
        (0.001, None), # V
        (None, None),  # k_1
        (None, None),  # p_1
        (None, None),  # k_2
        (None, None),  # p_2
        (None, None),  # epsilon -- unbounded on purpose, see normalised()
        (None, None),  # E
    ))
    name = 'Koniferka'

    def __init__(self,
                 x0: float = 0, y0: float = 0, a0: float = 0,
                 A: float = 0, F: float = 0,
                 V: float = 1, p1: float = 0, r1: float = np.inf, p2: float = 0, r2: float = np.inf,
                 epsilon: float = 0, E: float = 0):
        super().__init__()
        self.axis_shifter = TiltShifter(x0=x0, y0=y0, a0=a0, A=A, F=F, E=E)
        self.radial_transform = SaneBiexponentialTransformer(V, p1, r1, p2, r2)
        self.zenith_shifter = ZenithShifter(epsilon=epsilon, E=E)

    def __call__(self,
                 x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        r, b = self.axis_shifter(x, y)
        u = self.radial_transform(r)
        z, a = self.zenith_shifter(u, b)
        return z, a

    def invert(self,
               z: NDArray[np.floating],
               a: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        u, b = self.zenith_shifter.invert(z, a)
        r = self.radial_transform.invert(u)
        x, y = self.axis_shifter.invert(r, b)
        return x, y

    def __str__(self):
        return f"Koniferka projection with \n" \
               f"   {self.axis_shifter} \n" \
               f"   {self.radial_transform} \n" \
               f"   {self.zenith_shifter}"

    def as_dict(self):
        return self.axis_shifter.as_dict() | self.radial_transform.as_dict() | self.zenith_shifter.as_dict()

    def as_tuple(self):
        return (
            self.axis_shifter.x0, self.axis_shifter.y0, self.axis_shifter.a0,
            self.axis_shifter.A, self.axis_shifter.F,
            self.radial_transform.V,
            self.radial_transform.p1, self.radial_transform.r1,
            self.radial_transform.p2, self.radial_transform.r2,
            self.zenith_shifter.epsilon, self.zenith_shifter.E,
        )

    def normalised(self) -> 'KoniferkaProjection':
        """
        The same projection with its angles wrapped into range. See base.normalise_angles.

        Unpacked by name rather than by index: this class and its neighbours happen to carry their
        angles at the same positions of as_tuple(), and relying on that is how a parameter ends up
        read from the wrong slot.
        """
        x0, y0, a0, A, F, V, p1, r1, p2, r2, epsilon, E = self.as_tuple()
        a0, F, epsilon, E = normalise_angles(a0, F, epsilon, E)
        return KoniferkaProjection(x0, y0, a0, A, F, V, p1, r1, p2, r2, epsilon, E)

    @classmethod
    def from_dotmap(cls, dm):
        return cls(
            dm.x0, dm.y0, dm.a0,
            dm.A, dm.F,
            dm.V, dm.p1, dm.r1, dm.p2, dm.r2,
            dm.epsilon, dm.E,
        )
