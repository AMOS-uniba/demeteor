import datetime

import numpy as np
import pandas as pd

from pathlib import Path

from astropy.coordinates import EarthLocation, FK5, SkyCoord, AltAz, get_body, concatenate, Angle
from astropy.time import Time
from astropy import units as u


class Catalogue:
    def __init__(self, filename: Path = None):
        self._populated = False

        if filename is not None:
            self._populated = True
            self.planets = []
            self.planets_skycoord = None
            self.stars = pd.read_csv(filename, sep='\t', header=1)
            self.stars_skycoord = SkyCoord(self.stars.ra.to_numpy() * u.deg,
                                           self.stars.dec.to_numpy() * u.deg,
                                           frame=FK5(equinox=Time('J2000')))

    def build_planets(self,
                      location: EarthLocation,
                      time: Time = None):
        if time is None:
            time = Time(datetime.datetime.now(tz=datetime.UTC))

        sun = get_body('sun', time=time, location=location)

        planets = pd.DataFrame(columns=self.stars.columns)

        index = len(self.stars)
        for name in ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']:
            body = get_body(name, time=time, location=location)
            sundist = body.hcrs.distance
            phase = body.separation(sun)
            new_planet = pd.DataFrame(
                data=[
                    [
                        body.ra.degree,
                        body.dec.degree,
                        body.distance.to(u.lightyear).value,
                        self.planet_brightness(name, body.distance, sundist, phase),
                        -10
                    ]
                ],
                columns=self.stars.columns,
                index=[index]
            )
            if len(planets) > 0:
                planets = pd.concat([planets, new_planet])
                index += 1
            else:
                planets = new_planet

        self.planets = planets
        self.planets_skycoord = SkyCoord(planets.ra.to_numpy() * u.deg,
                                         planets.dec.to_numpy() * u.deg,
                                         frame=FK5(equinox=Time('J2000')))

    def altaz(self,
              location: EarthLocation,
              time: Time = None,
              *,
              planets: bool = True) -> AltAz:
        """
        Return the catalogue in alt-az coordinates at `location` and at `time`.
        Optionally include planets.
        """
        if self._populated:
            if time is None:
                time = Time(datetime.datetime.now(tz=datetime.UTC))

            altaz = AltAz(location=location, obstime=time, pressure=100000 * u.pascal, obswl=550 * u.nm)
            if planets:
                self.build_planets(location, time)
                total = concatenate([self.stars_skycoord, self.planets_skycoord])
                return total.transform_to(altaz)
            else:
                return self.stars_skycoord.transform_to(altaz)
        else:
            return AltAz()

    def vmag(self,
             location: EarthLocation,
             time: Time = None) -> np.ndarray[float]:
        """
        Return visual magnitudes of all objects at `location` and at `time`.
        Optionally include planets.
        """
        self.build_planets(location, time)
        return pd.concat([self.stars, self.planets]).vmag.to_numpy()

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
        return len(self.stars) + 7