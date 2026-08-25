import copy
import datetime
import logging
import dotmap
import numpy as np

from typing import Optional
from astropy.coordinates import EarthLocation
import astropy.units as u

from demeteor.projections.shifters import ScalingShifter
from .dotcollection import DotCollection
from .rect import Rect

log = logging.getLogger('demeteor')


class SensorData:
    """ A set of stars and meteor snapshots in xy format """

    def __init__(self,
                 stars: Optional[DotCollection] = None,
                 meteor: Optional[DotCollection] = None,
                 *,
                 location: Optional[EarthLocation] = None,
                 timestamp: Optional[datetime.datetime] = None,
                 name: str = "(unknown)",
                 station: Optional[str] = None,
                 bounds: Optional[Rect] = None,
                 fps: int = 1):
        self.rect = Rect(-1, 1, -1, 1) if bounds is None else bounds
        self.shifter = ScalingShifter(x0=self.rect.xcen, y0=self.rect.ycen, xs=0.0044, ys=0.0044)

        self._stars_raw = DotCollection() if stars is None else stars
        self._stars_scaled = DotCollection() if stars is None else stars
        self._meteor_raw = DotCollection() if meteor is None else meteor
        self._meteor_scaled = DotCollection() if meteor is None else meteor
        self.name = "(unknown)" if name is None else name,
        self.fps = fps
        self.station = "(unknown station)" if station is None else station

        self.location = location
        self.timestamp = datetime.datetime.now() if timestamp is None else timestamp

    # There was a load_YAML() here, reading the Kvant sighting format. It stayed behind in vasco
    # for now: a reader for a station's file format is a different thing from a container for what
    # it holds, and there are two hand-written readers for that format in this ecosystem which
    # want unifying deliberately rather than by accident. See the note in vasco/io.py.

    def set_shifter_scales(self, xs, ys):
        self.shifter.xs = xs
        self.shifter.ys = ys

        self.rescale_stars()
        self.rescale_meteor()
        log.debug(f"Set shifter scales to xs = {xs:.6f} mm, ys = {ys:.6f} mm")

    def rescale_stars(self):
        self._stars_scaled = DotCollection(
            np.stack(self.shifter(self._stars_raw.xs(masked=False), self._stars_raw.ys(masked=False)), axis=1),
            self._stars_raw.intensities(masked=False),
            mask=self._stars_raw.mask,
        )

    def rescale_meteor(self):
        self._meteor_scaled = DotCollection(
            np.stack(self.shifter(self._meteor_raw.xs(masked=False), self._meteor_raw.ys(masked=False)), axis=1),
            self._meteor_raw.intensities(masked=False),
            fnos=self._meteor_raw.fnos(masked=False),
            mask=self._meteor_raw.mask,
        )

    def _collection_to_disk(self, collection, masked):
        return np.stack(self.shifter(collection.xs(masked), collection.ys(masked)), axis=1)

    def stars_to_disk(self, masked):
        return self._collection_to_disk(self.stars, masked)

    def meteor_to_disk(self, masked):
        return self._collection_to_disk(self.meteor, masked)

    def reset_mask(self):
        self._stars_scaled.mask = None

    @property
    def stars_raw(self):
        """ Stars in raw (pixel) coordinates, as detected """
        return self._stars_raw

    @property
    def stars(self):
        """ Stars in scaled (mm) coordinates """
        return self._stars_scaled

    @property
    def meteor_raw(self):
        """ Meteor in raw (pixel) coordinates, as detected """
        return self._meteor_raw

    @property
    def meteor(self):
        """ Meteor in scaled (mm) coordinates """
        return self._meteor_scaled

    def __str__(self):
        return f"<Sensor data with {self.stars.count_visible} / {self.stars.count} " \
               f"reference objects and {self.meteor.count} meteor snapshots>"
