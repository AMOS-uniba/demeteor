import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from pathlib import Path

from astropy.coordinates import EarthLocation, FK5, SkyCoord, AltAz, get_body, concatenate, Angle
from astropy.time import Time
from astropy import units as u
from numpy.ma.core import shape


class Catalogue:
    COLUMNS = ['ra', 'dec', 'dist', 'vmag', 'absmag']
    PLANETS = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

    def __init__(self, filename: Path = None):
        self.planets = []
        self.planets_skycoord = None

        self.build_planets()

        if filename is None:
            self._populated = False
            self.stars = pd.DataFrame(columns=self.COLUMNS)
            self.stars_skycoord = SkyCoord([] * u.deg,
                                           [] * u.deg,
                                           frame=FK5(equinox=Time('J2000')))
        else:
            self._populated = True
            self.stars = pd.read_csv(filename, sep='\t', header=1)
            self.stars_skycoord = SkyCoord(self.stars.ra.to_numpy() * u.deg,
                                           self.stars.dec.to_numpy() * u.deg,
                                           frame=FK5(equinox=Time('J2000')))

        assert self.stars.columns.tolist() == self.COLUMNS, \
            f"Invalid format for columns, expected {self.COLUMNS}"

        self._mask = np.ones(shape=len(self.stars) + len(self.planets), dtype=bool)

    @property
    def populated(self) -> bool:
        return self._populated

    def build_planets(self,
                      location: EarthLocation = EarthLocation.from_geodetic(0, 0, 0),
                      time: Time = None):
        if time is None:
            time = Time(datetime.datetime.now(tz=datetime.UTC))

        sun = get_body('sun', time=time, location=location)

        planets = []
        for name in self.PLANETS:
            body = get_body(name, time=time, location=location)
            sundist = body.hcrs.distance
            phase = body.separation(sun)
            planets.append(pd.DataFrame(
                data=[
                    [
                        body.ra.degree,
                        body.dec.degree,
                        body.distance.to(u.lightyear).value,
                        self.planet_brightness(name, body.distance, sundist, phase),
                        -10
                    ]
                ],
                columns=self.COLUMNS,
            ))

        self.planets = pd.concat(planets)
        self.planets_skycoord = SkyCoord(self.planets.ra.to_numpy() * u.deg,
                                         self.planets.dec.to_numpy() * u.deg,
                                         frame=FK5(equinox=Time('J2000')))

    def radec(self,
              location: EarthLocation,
              time: Time = None,
              *,
              planets: bool = True,
              masked: bool) -> SkyCoord:
        if time is None:
            time = Time(datetime.datetime.now(tz=datetime.UTC))

        if planets:
            self.build_planets(location, time)
            total = np.concatenate([self.planets_skycoord, self.stars_skycoord])
        else:
            total = self.stars_skycoord

        return total[self.mask] if masked else total

    def altaz(self,
              location: EarthLocation,
              time: Time = None,
              *,
              planets: bool = True,
              masked: bool) -> AltAz:
        """
        Return the catalogue in alt-az coordinates at `location` and at `time`.
        Optionally include planets.
        """
        if time is None:
            time = Time(datetime.datetime.now(tz=datetime.UTC))

        altaz = AltAz(location=location, obstime=time, pressure=100000 * u.pascal, obswl=550 * u.nm)
        radec = self.radec(location, time, planets=planets, masked=masked)
        return radec.transform_to(altaz)

    def vmag(self,
             location: EarthLocation,
             time: Time = None,
             *,
             masked: bool) -> np.ndarray[float]:
        """
        Return visual magnitudes of all objects at `location` and at `time`.
        Optionally include planets.
        """
        self.build_planets(location, time)
        vmags = pd.concat([self.planets, self.stars]).vmag.to_numpy(dtype=float)
        return vmags[self.mask] if masked else vmags

    @staticmethod
    def planet_brightness(planet: str,
                          distance_earth: u.Quantity,
                          distance_sun: u.Quantity,
                          phase: Angle):
        """
        Get the approximate visual magnitude of a planet.
        Shamelessly stolen from APC, Montenbruck 1999
    """
        p = phase.degree / 100.0

        match planet:
            case 'mercury':
                mag = -0.42 + (3.80 - (2.73 - 2 * p) * p) * p
            case 'venus':
                mag = -4.40 + (0.09 + (2.39 - 0.65 * p) * p) * p
            case 'mars':
                mag = -1.52 + 1.6 * p
            case 'jupiter':
                mag = -9.4 + 0.5 * p
            case 'saturn':
                # Currently we do not care about the rings, but it might be worth checking it later
                sd = 0 # np.abs(np.sin(lat))
                dl = 0 # np.abs((dlong + np.pi) % (2 * np.pi) - np.pi) / 100
                mag = -8.88 + 2.60 * sd + 1.25 * sd**2 + 4.4 * dl
            case 'uranus':
                mag = -7.19
            case 'neptune':
                mag = -6.87

        return mag + 5 * np.log10(distance_earth.to(u.au).value * distance_sun.to(u.au).value)

    @property
    def count(self) -> int:
        return len(self.stars) + len(self.planets)

    @property
    def count_visible(self) -> int:
        return len(self.mask[self.mask])

    @property
    def mask(self) -> np.ndarray[bool]:
        return self._mask

    @mask.setter
    def mask(self, m: Optional[np.ndarray[bool]] = None) -> None:
        self._mask = np.ones(shape=(self.count,), dtype=bool) if m is None else m
        assert self.mask.shape == (self.count,), \
            f"Mask shape does not match data shape: expected {self.count,}, got {self.mask.shape}"

    def __str__(self):
        return f"<Catalogue with {self.count_visible} / {self.count} reference objects>"
