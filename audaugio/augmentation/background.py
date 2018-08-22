import numpy as np

from .augmentation_base import AugmentationBase


class BackgroundNoiseAugmentation(AugmentationBase):
    """
    Add background noise randomly sampled from a 0-centered normal distribution.
    """

    def __init__(self, amplitude):
        """
        :param amplitude: desired amplitude of the noise, equivalent to the standard deviation of the distribution
        """
        super().__init__(replaces=False)
        self.amplitude = amplitude

    def augment(self, signal, sr):
        noise = np.random.normal(0, self.amplitude, len(signal))
        return [np.array(signal) + noise]
