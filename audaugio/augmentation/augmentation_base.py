import os

import librosa
import sox
from sox import SoxError


class AugmentationBase:
    """
    Base class for an augmentation. Implement this to create your own augmentations.

    :param replaces: whether the augmentation should replace the audio it augments. Usually will be false.
    """

    def __init__(self, replaces: bool, *kwargs):
        self.replaces = replaces

    def augment(self, signal, sr):
        """
        Given a signal, apply the augmentation and return all resulting augmented audio as a list (even if it's a single signal).

        :param signal: unaugmented signal, ndarray
        :param sr: sample rate, int
        """
        raise NotImplementedError


class SoxAugmentationBase(AugmentationBase):
    """
    Base class for an augmentation that depends on SoX. Because all SoX transformations are applied in the same way, inheritors of this class only need to
    set self.transformer in their __init__() method.

    :param replaces: whether the augmentation should replace the audio it augments. Usually will be false.
    """

    def __init__(self, replaces: bool, *kwargs):
        super().__init__(replaces, *kwargs)
        self.transformer = sox.Transformer()
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
