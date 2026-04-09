import numpy as np

from . import constants

from astropy import units as u
from astropy.coordinates import Angle


class AirMass:
    def kasten_young(altitude: Angle,
                     elevation: u.Quantity = 0 * u.m):
        return np.where(
            altitude >= 0 * u.deg,
            AirDensity.isa(elevation) / AirDensity.isa(0 * u.m) / (np.sin(altitude) + 0.50572 * ((altitude.degree + 6.07995) ** (-1.6364))),
            np.inf
        )

    def pickering2002(altitude: u.Quantity,
                      elevation: u.Quantity = 0 * u.m):
        return np.where(
            altitude >= 0 * u.deg,
            AirDensity.isa(elevation) / AirDensity.isa(0 * u.m) / (np.sin(altitude + np.radians(244 / (165 + 47 * altitude.degree ** 1.1)))),
            np.inf
        )

    def attenuate(flux: u.Quantity,
                  air_mass: u.Quantity,
                  one: float = constants.AttenuationOneAirMassMag):
        return np.where(air_mass <= 100, flux * np.exp(-np.log(100) / 5 * one * air_mass), 0)


class AirDensity:
    def isa(altitude: u.Quantity):
        return 101325 * np.exp(-altitude.to(u.m) / (7990 * u.m)) * u.pascal
