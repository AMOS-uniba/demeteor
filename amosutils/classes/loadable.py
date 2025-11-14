from abc import ABC, abstractmethod
from typing import Any


class Loadable(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def from_dict(cls,
                  config: dict[str, Any]):
        """
        Load from a dictionary. This expects that its 'name' attribute points to the actual class in the registry.
        """
        return cls._registry[config['name']](**config['parameters'])

    def as_dict(self):
        """ Return a dict representation of the Projection's parameters """
        return {
            'name': self.__class__.__name__,
        }

    @abstractmethod
    def _as_dict(self):
        pass
