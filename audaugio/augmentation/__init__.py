import os

import librosa
import numpy as np
import sox
from sox import SoxError


class Augmentation:
    def __init__(self, replaces: bool, *kwargs):
        """
        Base class for an augmentation. Implement this to create your own augmentations.

        :param replaces: whether the augmentation should replace the audio it augments. Usually will be false.
        """
        self.replaces = replaces

    def augment(self, signal, sr):
        """
        Given a signal, apply the augmentation and return all resulting augmented audio as a list (even if it's a single signal).

        :param signal: unaugmented signal, ndarray
        :param sr: sample rate, int
        """
        raise NotImplementedError


class BackgroundNoiseAugmentation(Augmentation):
    def __init__(self, amplitude):
        """
        Add background noise randomly sampled from a 0-centered normal distribution.

        :param amplitude: desired amplitude of the noise, equivalent to the standard deviation of the distribution
        """
        super().__init__(replaces=False)
        self.amplitude = amplitude

    def augment(self, signal, sr):
        noise = np.random.normal(0, self.amplitude, len(signal))
        return [np.array(signal) + noise]


class EqualizerAugmentation(Augmentation):
    def __init__(self, frequency: float, resonance: float, gain: float):
        """
        Add an arbitrarily tall and wide frequency filter at an arbitrary frequency.

        :param frequency: center of the filter
        :param resonance: width of the filter as a q-factor
        :param gain: height of the filter in dB
        """
        super().__init__(replaces=False)
        self.transformer = sox.Transformer()
        self.transformer.equalizer(frequency, resonance, gain)
        self.input_file = './temporary_augmented_audio_in.wav'
        self.output_file = './temporary_augmented_audio_out.wav'

    def augment(self, signal, sr):
        librosa.output.write_wav(self.input_file, signal, sr=sr)
        try:
            self.transformer.build(self.input_file, self.output_file)
        except SoxError as e:
            self.cleanup()
            if 'sox: command not found' in e.args[0]:
                raise OSError("You need a working installation of SoX to use this augmentation.\nIf you haven't installed it, "
                              "go to http://sox.sourceforge.net/ for a download link. Otherwise, double check your path variables.")
            else:
                raise e
        signal, sr = librosa.load(self.output_file, sr=sr)
        self.cleanup()
        return [signal]

    def cleanup(self):
        try:
            os.remove(self.input_file)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.output_file)
        except FileNotFoundError:
            pass


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


class WindowingAugmentation(Augmentation):
    def __init__(self, window_length: float, hop_size: float, drop_last=False):
        """
        Window a signal into many segments of equal length. If hop_size is less than window_length, these windows will overlap.

        :param window_length: the length in seconds of a window
        :param hop_size: the distance in seconds between the start of each window
        :param drop_last: whether to zero-pad the last segment of audio when it is shorter than window_length. If false, this part of the signal is dropped.
        """
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
