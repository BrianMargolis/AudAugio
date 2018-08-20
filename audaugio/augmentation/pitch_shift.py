import librosa

from audaugio.augmentation import Augmentation


class PitchShiftAugmentation(Augmentation):
    def __init__(self, steps):
        """
        Pitch shift a signal by half-steps without changing the duration.

        :param steps:
        """
        super().__init__(replaces=False)
        self.steps = steps

    def augment(self, signal, sr):
        return [librosa.effects.pitch_shift(signal, sr, self.steps)]
