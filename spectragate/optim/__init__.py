# SpectraGate AGPL-3.0 License - https://ultralytics.com/license

"""
SpectraGate optimizer module.

This module provides custom optimizers for neural network training,
including the MuSGD hybrid optimizer combining Muon and SGD.
"""

from .muon import Muon, MuSGD

__all__ = ["MuSGD", "Muon"]
