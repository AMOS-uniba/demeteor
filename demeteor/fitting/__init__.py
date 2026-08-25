"""
Fitting a plate to reference dots, start to finish.

`fit(job)` takes a mapping describing one frame -- where and when, the dots, the plate to start
from, and how hard to try -- and gives back the reductions it arrived at. Everything between is
policy that neither a window nor a web server should own a private copy of: masking the dots the
plate puts below the horizon, dropping catalogue stars that no dot came near, sigma-clipping the
aeroplane that a person would have masked by hand, and choosing a smoothing bandwidth by
cross-validation instead of by guess.
"""
from .driver import JOB_FORMAT, REDUCTION_FORMAT, RESULT_FORMAT, JobError, fit

__all__ = ['JOB_FORMAT', 'REDUCTION_FORMAT', 'RESULT_FORMAT', 'JobError', 'fit']
