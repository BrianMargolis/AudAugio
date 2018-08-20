import unittest

import numpy as np

from audaugio.augmentation_chain import AugmentationChain
from audaugio.augmentors.background_noise import BackgroundNoiseAugmentation
from audaugio.augmentors.equalizer import EqualizerAugmentation
from audaugio.augmentors.pitch_shift import PitchShiftAugmentation
from audaugio.augmentors.time_stretch import TimeStretchAugmentation


class TestAugmentationChain(unittest.TestCase):
    def setUp(self):
        # mock a random signal
        self.len_sec = np.random.randint(low=5, high=20)
        self.sr = np.random.randint(low=16000, high=44000)
        self.signal = self.mock_audio(self.len_sec, self.sr)
        range_ = [self.mock_audio(np.random.randint(low=5, high=20), self.sr) for i in range(20)]
        self.signals = np.array(range_)

    def test_empty_chain(self):
        chain = AugmentationChain()
        augmented = chain(self.signal, self.sr)
        self.assertEqual(len(augmented), 1)
        self.assertEqual(np.sum(augmented[0] == self.signal), len(augmented[0]))

    def test_small_chain(self):
        chain = AugmentationChain(PitchShiftAugmentation(1),
                                  TimeStretchAugmentation(.95),
                                  EqualizerAugmentation(400, 1, 2))
        augmented = chain(self.signal, self.sr)
        self.assertEqual(len(augmented), 2 ** 3)

    def test_large_chain(self):
        chain = AugmentationChain(PitchShiftAugmentation(-1),
                                  TimeStretchAugmentation(.95),
                                  EqualizerAugmentation(300, 1, 2),
                                  BackgroundNoiseAugmentation(.005),
                                  TimeStretchAugmentation(1.25))
        augmented = chain(self.signal, self.sr)
        self.assertEqual(len(augmented), 2 ** 5)

    @staticmethod
    def mock_audio(len_sec, sr):
        return np.random.random(size=sr * len_sec)


if __name__ == '__main__':
    unittest.main()
