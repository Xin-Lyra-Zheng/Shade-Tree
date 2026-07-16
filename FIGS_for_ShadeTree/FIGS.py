"""Local FIGS compatibility entry point.

The implementation is supplied by the installed ``imodels`` package, which is
already a runtime dependency of the original project copy of FIGS.
"""

from imodels import FIGSClassifier, FIGSRegressor

__all__ = ["FIGSClassifier", "FIGSRegressor"]
