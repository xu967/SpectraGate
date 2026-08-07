# SpectraGate 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""兼容入口：旧路径保留。

请改用：spectragate.models.rtdetrmm.predict.RTDETRMMPredictor
"""

from spectragate.models.rtdetrmm.predict import RTDETRMMPredictor

__all__ = ["RTDETRMMPredictor"]
