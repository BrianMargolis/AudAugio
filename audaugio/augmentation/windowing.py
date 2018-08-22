import numpy as np

from .augmentation_base import AugmentationBase


class WindowingAugmentation(AugmentationBase):
    """
    Window a signal into many segments of equal length. If hop_size is less than window_length, these windows will overlap.

    :param window_length: the length in seconds of a window
    :param hop_size: the distance in seconds between the start of each window
    :param drop_last: whether to zero-pad the last segment of audio when it is shorter than window_length. If false, this part of the signal is dropped.
    """

    def __init__(self, window_length: float, hop_size: float, drop_last=False):
        super().__init__(replaces=True)
        self.window_length = window_length
        self.hop_size = hop_size
        self.drop_last = drop_last

    def augment(self, signal, sr):
        window_samples = self.window_length * sr
        hop_samples = self.hop_size * sr
        audio_samples = signal.shape[0]

        if window_samples >= audio_samples:
            return [signal]

        segments = []
        start = 0
        while start + window_samples <= audio_samples:
            segments.append(signal[start:start + window_samples])
            start += hop_samples

        if start - hop_samples + window_samples < audio_samples and not self.drop_last:
            pad = np.zeros(window_samples - (audio_samples - start))
            last_segment = np.append(signal[start:], pad)
            segments.append(last_segment)

        return segments
