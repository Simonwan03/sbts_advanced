"""
Backward-compatible visualization exports.

The active plotting code now lives under the ``visualization`` package.
This wrapper keeps older imports working for notebooks and legacy scripts.
"""

from visualization.general import *  # noqa: F401,F403
from visualization.general import _calc_autocorr  # noqa: F401

