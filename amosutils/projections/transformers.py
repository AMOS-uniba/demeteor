from abc import ABC, abstractmethod
import numpy as np
import scipy as sp


class RadialTransformer(ABC):
    """ Class for transforming radial distances in all-sky projections """

    def __call__(self, r):
        raise NotImplementedError("Radial transformers must implement __call__(r: np.ndarray) -> np.ndarray")

    @abstractmethod
    def fprime(self, u):
        """ Derivative function for the Newton method, not implemented in the abstract base class """
        raise NotImplementedError("Radial transformers must implement df/dr as fprime(u: np.ndarray) -> np.ndarray")

    def invert(self, u):
        """ Numerically approximate the inverse function using the Newton method """
        return sp.optimize.newton(lambda r: self.__call__(r) - u, np.zeros_like(u), self.fprime)


class LinearTransformer(RadialTransformer):
    """ Linear radial transform, u = Vr """

    def __init__(self, V: float = 1):
        assert V > 0, "Radial linear scale V must be > 0"
        self.V = V  # radial stretch, linear coefficient

    def __call__(self, r):
        return self.V * r

    def fprime(self, r):
        """ du / dr = V """
        return self.V * np.ones_like(r)

    def as_dict(self):
        return dict(
            V=float(self.V),
        )


class ExponentialTransformer(LinearTransformer):
    """ Linear + exponential radial correction, u = Vr + S(e^(Dr) - 1) """

    def __init__(self, V: float = 1, S: float = 0, D: float = 0):
        super().__init__(V)
        self.S = S  # radial stretch, exponential term, linear coefficient
        self.D = D  # radial stretch, exponential term, exponent coefficient

    def __call__(self, r):
        return super().__call__(r) + self.S * (np.exp(self.D * r) - 1)

    def fprime(self, r):
        """ du/dr = V + SDe^(Dr) """
        return super().fprime(r) + self.S * self.D * np.exp(self.D * r)

    def as_dict(self):
        return super().as_dict() | dict(
            S=float(self.S),
            D=float(self.D),
        )


class BiexponentialTransformer(ExponentialTransformer):
    """ Bi-exponential radial fitting procedure, u = Vr + S(e^(Dr) - 1) + P(e^(Qr^2) - 1) """

    def __init__(self, V: float = 0,
                 S: float = 0, D: float = 0,
                 P: float = 0, Q: float = 0):
        super().__init__(V, S, D)
        self.P = P # radial stretch, square-exponential term, linear coefficient
        self.Q = Q # radial stretch, square-exponential term, exponent coefficient

    def __call__(self, r):
        return super().__call__(r) + self.P * (np.exp(self.Q * r * r) - 1)

    def fprime(self, r):
        """ du/dr = V + SDe^(Dr) + 2 PQr e^(Qr^2) """
        return super().fprime(r) + 2 * self.P * self.Q * r * np.exp(self.Q * r * r)

    def __str__(self):
        return f"<{self.__class__.__name__} {self.V=} {self.S=} {self.D=} {self.P=} {self.Q=}>"

    def as_dict(self):
        return super().as_dict() | dict(
            P=float(self.P),
            Q=float(self.Q),
        )


class SaneExponentialTransformer(LinearTransformer):
    """ Sanitized linear + exponential radial correction, u = Vr + k_1 * (e^(r/r_1) - 1) """

    def __init__(self, V: float = 1, p1: float = 0, r1: float = 0):
        super().__init__(V)
        assert abs(r1) > 0.1, "Exponent scale r1 must be > 0.1"
        self.p1 = p1  # radial stretch, exponential term, linear coefficient
        self.r1 = r1    # radial stretch, exponential term, exponent coefficient

    def __call__(self, r):
        return super().__call__(r) + self.p1 * (np.exp(r / self.r1) - 1)

    def fprime(self, r):
        """ du/dr = V + S / r_1 * e^(r / r_1) """
        return super().fprime(r) + self.p1 / self.r1 * np.exp(r / self.r1)

    def as_dict(self):
        return super().as_dict() | dict(
            p1=float(self.p1),
            r1=float(self.r1),
        )


class SaneBiexponentialTransformer(SaneExponentialTransformer):
    def __init__(self, V: float = 0,
                 p1: float = 0, r1: float = np.inf,
                 p2: float = 0, r2: float = np.inf):
        super().__init__(V, p1, r1)
        assert abs(r2) > 0.1, "Exponent scale r2 must be > 0.1 mm"
        self.p2 = p2  # radial stretch, square-exponential term, linear coefficient
        self.r2 = r2  # radial stretch, square-exponential term, exponent coefficient

    def __call__(self, r):
        return super().__call__(r) + self.p2 * (np.exp((r / self.r2)**2) - 1)

    def fprime(self, r):
        """ du/dr = V + SDe^(Dr) + 2 PQr e^(Qr^2) """
        return super().fprime(r) + 2 * self.p2 / self.r2**2 * r * np.exp((r / self.r2)**2)

    def __str__(self):
        return f"<{self.__class__.__name__} {self.V=} {self.p1=} {self.r1=} {self.p2=} {self.r2=}>"

    def as_dict(self):
        return super().as_dict() | dict(
            p2=float(self.p2),
            r2=float(self.r2),
        )


