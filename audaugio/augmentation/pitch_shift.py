import librosa

from .augmentation_base import AugmentationBase


class PitchShiftAugmentation(AugmentationBase):
    """
    Pitch shift a signal by half-steps without changing the duration.
    """

    def __init__(self, steps):
        super().__init__(replaces=False)
        self.steps = steps

    def augment(self, signal, sr):
        return [librosa.effects.pitch_shift(signal, sr, self.steps)]
