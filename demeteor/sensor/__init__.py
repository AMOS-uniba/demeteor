"""
What a camera saw, before any of it is believed.

A `SensorData` is one frame's worth of measurements in sensor coordinates: the reference stars its
software found and the trail it thought was a meteor, each a `DotCollection` of positions and
intensities. Nothing here knows where the camera was pointing or what the dots are -- that is the
matcher's business, and `disk` is the coordinate system the two meet in.

Sensor coordinates are millimetres, not pixels, because a plate is a map from millimetres. Whoever
knows the pixel size does the scaling; `SensorData` will do it through a ScalingShifter if asked
and otherwise leaves the numbers exactly as handed in.
"""
from .disk import (QuarterTau, altaz_to_disk, disk_to_altaz, mask_sparse, numpy_to_disk,
                   proj_to_disk)
from .dotcollection import DotCollection
from .rect import Rect
from .sensordata import SensorData

__all__ = ['DotCollection', 'QuarterTau', 'Rect', 'SensorData', 'altaz_to_disk', 'disk_to_altaz',
           'mask_sparse', 'numpy_to_disk', 'proj_to_disk']
