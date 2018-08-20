import os

import librosa
import sox

from audiaug.augmentors.generics import Augmentation


class EqualizerAugmentation(Augmentation):
    def __init__(self, frequency: float, resonance: float, gain: float):
        super().__init__(replaces=False)
        self.transformer = sox.Transformer()
        self.transformer.equalizer(frequency, resonance, gain)
        self.input_file = './temporary_augmented_audio_in.wav'
        self.output_file = './temporary_augmented_audio_out.wav'

    def augment(self, audio, sr):
        librosa.output.write_wav(self.input_file, audio, sr=sr)
        self.transformer.build(self.input_file, self.output_file)
        audio, sr = librosa.load(self.output_file, sr=sr)
        self.cleanup()
        return [audio]

    def cleanup(self):
        os.remove(self.input_file)
        os.remove(self.output_file)
