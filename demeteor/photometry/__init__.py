"""
Turning raw sensor intensities into magnitudes, and back.

A zero point and a logarithm, which is as much as an uncalibrated all-sky frame supports: there are
no dark frames here and no flat field, so the scale is anchored by one number and the rest is the
definition of a magnitude.
"""
from .calibration import Calibration, LogCalibration

__all__ = ['Calibration', 'LogCalibration']
