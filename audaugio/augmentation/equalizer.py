from .augmentation_base import SoxAugmentationBase


class EqualizerAugmentation(SoxAugmentationBase):
    """
    Add an arbitrarily tall and wide frequency filter at an arbitrary frequency.

    :param frequency: center of the filter
    :param resonance: width of the filter as a q-factor
    :param gain: height of the filter in dB
    """

    def __init__(self, frequency: float, resonance: float, gain: float):
        super().__init__(replaces=False)
        self.transformer.equalizer(frequency, resonance, gain)


class LowPassAugmentation(SoxAugmentationBase):
    """
    Filter out high frequencies.

    :param frequency: center of the filter
    :param resonance: width of the filter as a q-factor. Only applies when n_poles = 2.
    :param n_poles: either 1 or 2. Number of poles in the filter.
    """
    def __init__(self, frequency: float, resonance: float, n_poles: int = 1):
        super().__init__(replaces=False)
        if n_poles not in [1, 2]:
            raise ValueError("n_poles must be 1 or 2, not {0}".format(n_poles))
        self.transformer.lowpass(frequency, resonance, n_poles)


class HighPassAugmentation(SoxAugmentationBase):
    """
    Filter out low frequencies.

    :param frequency: center of the filter
    :param resonance: width of the filter as a q-factor. Only applies when n_poles = 2.
    :param n_poles: either 1 or 2. Number of poles in the filter.
    """
    def __init__(self, frequency: float, resonance: float, n_poles: int = 1):
        super().__init__(replaces=False)
        if n_poles not in [1, 2]:
            raise ValueError("n_poles must be 1 or 2, not {0}".format(n_poles))
        self.transformer.highpass(frequency, resonance, n_poles)
