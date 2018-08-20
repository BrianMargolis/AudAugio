import librosa

from audaugio.augmentation import Augmentation


class TimeStretchAugmentation(Augmentation):
    def __init__(self, rate):
        """
        Change the duration of a signal without changing its pitch.

        :param rate: factor by which to speed up or slow down the signal. When rate is 1, the signal is not modified.
        """
        super().__init__(replaces=False)
        self.rate = rate

    def augment(self, signal, sr):
        return [librosa.effects.time_stretch(signal, self.rate)]
