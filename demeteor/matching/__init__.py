"""
Deciding which dot is which star, and fitting a plate so that they agree.

A `Matcher` holds three things -- a catalogue, one frame's sensor data, and a projection -- and the
pairing between the first two that the third implies. Change the projection and the pairing changes;
change the pairing and the residuals change; minimise the residuals and you have a fitted plate.

It is stateful on purpose, because a person driving a GUI turns one parameter at a time and wants to
see the effect. That also makes it the wrong shape for a library, and it is on the list: the mutable
match state and the fitting operations want separating, and `_calibration` wants to be an argument
rather than an attribute. Moved as it was so that every existing caller kept working -- the window,
the plots and all three exporters hold one of these.
"""
from .matcher import Matcher, NothingToPair

__all__ = ['Matcher', 'NothingToPair']
