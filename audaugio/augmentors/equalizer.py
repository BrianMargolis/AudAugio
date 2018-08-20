import os

import librosa
import sox
from sox import SoxError

from audaugio.augmentors.generics import Augmentation


class EqualizerAugmentation(Augmentation):
    def __init__(self, frequency: float, resonance: float, gain: float):
        super().__init__(replaces=False)
        self.transformer = sox.Transformer()
        self.transformer.equalizer(frequency, resonance, gain)
        self.input_file = './temporary_augmented_audio_in.wav'
        self.output_file = './temporary_augmented_audio_out.wav'

    def augment(self, audio, sr):
        librosa.output.write_wav(self.input_file, audio, sr=sr)
        try:
            self.transformer.build(self.input_file, self.output_file)
        except SoxError as e:
            self.cleanup()
            if 'sox: command not found' in e.args[0]:
                raise OSError("You need a working installation of SoX to use this augmentation.\nIf you haven't installed it, "
                              "go to http://sox.sourceforge.net/ for a download link. Otherwise, double check your path variables.")
            else:
                raise e
        audio, sr = librosa.load(self.output_file, sr=sr)
        self.cleanup()
        return [audio]

    def cleanup(self):
        try:
            os.remove(self.input_file)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.output_file)
        except FileNotFoundError:
            pass
