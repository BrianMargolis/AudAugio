import librosa

from audaugio.augmentation import Augmentation


class PitchShiftAugmentation(Augmentation):
    def __init__(self, steps):
        super().__init__(replaces=False)
        self.steps = steps

    def augment(self, audio, sr):
        return [librosa.effects.pitch_shift(audio, sr, self.steps)]
