"""Fusion modules for the SpectraGate band-aware progressive fusion framework.

Exposes the three fusion modules used by the SpectraGate architecture:
LMSG (LFEM_D_V5), CPSE (NLFEM_D_V5), and CCSF (FDFTM_V6).
"""

from .fredft import LFEM_D_V5, NLFEM_D_V5, FDFTM_V6  # SpectraGate fusion modules

__all__ = (
    'LFEM_D_V5',  # LMSG - within-modality multi-band spectral gating
    'NLFEM_D_V5',  # CPSE - cross-modal conjugate cross-power spectrum enhancement
    'FDFTM_V6',   # CCSF - cross-modal conjugate-spectrum fusion to single stream
)
