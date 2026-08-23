import math

import dotmap
import numpy as np
from typing import Tuple, Any

from abc import ABC, abstractmethod

import yaml


TAU = 2 * math.pi


def normalise_angles(a0: float,
                     F: float,
                     epsilon: float,
                     E: float) -> tuple[float, float, float, float]:
    """
    Bring the angles of a tilt-and-zenith-shifted projection into the range they mean, in radians.

    A fit is unconstrained: scipy walks these wherever it likes and nothing wraps them afterwards,
    so the same plate can come back as a0 = 12.7 rad or epsilon = -0.014. Every rewrite here leaves
    the projection pointing at exactly the same sky.

    a0, F and E are azimuths and take a whole turn. epsilon is a true zenith distance and takes
    half of one: reflecting it through the axis turns its azimuth around, so E picks up the half
    turn that epsilon gives up. That half turn is not optional -- flipping the sign of epsilon
    without it moves the sky by roughly twice the zenith distance, which is 0.055 rad on a typical
    all-sky plate and 2 rad on a badly aligned one -- which is why this is a function of all four
    together and cannot be done one parameter at a time.
    """
    epsilon %= TAU
    if epsilon > math.pi:
        epsilon, half_turn = TAU - epsilon, math.pi
    else:
        half_turn = 0.0

    return a0 % TAU, F % TAU, epsilon, (E + half_turn) % TAU


class Projection(ABC):
    """
    A base class for all projections. Should implement xy -> za and za -> xy conversions.
    """
    bounds = np.array((
        (0, None),
    ))

    _registry = {}

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply this projection to an array of points: xy -> za """

    @abstractmethod
    def invert(self, z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply an inverse projection to an array of points: za -> xy """

    @abstractmethod
    def as_dict(self) -> dict[str, float]:
        """ Return a dict representation of the Projection's parameters """
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Projection._registry[cls.name] = cls

    def normalised(self):
        """
        An equivalent projection with its angles in the range they mean.

        The default is for projections that have no angles to normalise; anything built on a tilt
        shifter and a zenith shifter overrides it. Never mutates: a fit result is worth keeping as
        it came out, and the caller decides what to do with either form.
        """
        return self

    @classmethod
    def from_dict(cls,
                  config: dict[str, Any]):
        return cls._registry[config['name']](**config['parameters'])

    @classmethod
    def from_dotmap(cls, dm):
        """
        Load from a dotmap. Useful as an intermediate step when loading from YAML.
        """

    @classmethod
    def load(cls, file):
        data = dotmap.DotMap(yaml.safe_load(file), _dynamic=False)
        data = data.projection.parameters
        return cls.from_dotmap(data)
