import os

import librosa
import sox
from sox import SoxError

from .augmentation_base import AugmentationBase


class EqualizerAugmentation(AugmentationBase):
    """
    Add an arbitrarily tall and wide frequency filter at an arbitrary frequency.

    :param frequency: center of the filter
    :param resonance: width of the filter as a q-factor
    :param gain: height of the filter in dB
    """

    def __init__(self, frequency: float, resonance: float, gain: float):
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
