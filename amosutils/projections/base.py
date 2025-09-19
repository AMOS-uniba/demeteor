import numpy as np
from typing import Tuple, Any

from abc import ABC, abstractmethod


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
    def as_dict(self):
        """ Return a dict representation of the Projection's parameters """
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Projection._registry[cls.name] = cls

    @classmethod
    def from_dict(cls,
                  config: dict[str, Any]):
        return cls._registry[config['name']](**config['parameters'])

