# how we call each module to run a specific task
from .train import train_model
from .diagnose import Diagnose
from .detect import DetectLabel
from .detect import DetectFeature
__all__ = [
    'train_model', 'Diagnose', 'DetectLabel', 'DetectFeature'
]