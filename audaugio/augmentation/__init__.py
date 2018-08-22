from .augmentation_base import AugmentationBase
from .background import BackgroundNoiseAugmentation
from .equalizer import EqualizerAugmentation
from .pitch_shift import PitchShiftAugmentation
from .time_stretch import TimeStretchAugmentation
from .windowing import WindowingAugmentation

__all__ = ['AugmentationBase', 'BackgroundNoiseAugmentation', 'EqualizerAugmentation', 'PitchShiftAugmentation', 'TimeStretchAugmentation',
           'WindowingAugmentation']
