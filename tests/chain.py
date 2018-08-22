import unittest

import numpy as np

from audaugio.chain.linear import LinearChain
from audaugio.chain.combinatoric import CombinatoricChain
from audaugio.augmentation.equalizer import EqualizerAugmentation
from audaugio.augmentation.time_stretch import TimeStretchAugmentation
from audaugio.augmentation.pitch_shift import PitchShiftAugmentation
from audaugio.augmentation.background import BackgroundNoiseAugmentation


class TestChain(unittest.TestCase):
    def setUp(self):
        # mock a random signal
        self.len_sec = np.random.randint(low=5, high=20)
        self.sr = np.random.randint(low=16000, high=44000)
        self.signal = self.mock_audio(self.len_sec, self.sr)

    @staticmethod
    def mock_audio(len_sec, sr):
        return np.random.random(size=sr * len_sec)


class TestCombinatoricChain(TestChain):
    def setUp(self):
        super().setUp()

    def test_empty_chain(self):
        chain = CombinatoricChain()
        augmented = chain(self.signal, self.sr)
        self.assertEqual(len(augmented), 1)
        self.assertEqual(np.sum(augmented[0] == self.signal), len(augmented[0]))

    def test_small_chain(self):
        chain = CombinatoricChain(PitchShiftAugmentation(1),
                                  TimeStretchAugmentation(.95),
                                  EqualizerAugmentation(400, 1, 2))
        augmented = chain(self.signal, self.sr)
        self.assertEqual(len(augmented), 2 ** 3)

    def test_large_chain(self):
        chain = CombinatoricChain(PitchShiftAugmentation(-1),
                                  TimeStretchAugmentation(.95),
                                  EqualizerAugmentation(300, 1, 2),
                                  BackgroundNoiseAugmentation(.005),
                                  TimeStretchAugmentation(1.25))
        augmented = chain(self.signal, self.sr)
        self.assertEqual(len(augmented), 2 ** 5)


class TestLinearChain(TestChain):
    def setUp(self):
        super().setUp()

    def test_empty_chain(self):
        chain = LinearChain()
        augmented = chain(self.signal, self.sr)
        self.assertEqual(len(augmented), 1)
        self.assertEqual(np.sum(augmented[0] == self.signal), len(augmented[0]))

    def test_small_chain(self):
        chain = LinearChain(PitchShiftAugmentation(1),
                            TimeStretchAugmentation(.95),
                            EqualizerAugmentation(400, 1, 2))
        augmented = chain(self.signal, self.sr)
        self.assertEqual(len(augmented), 1)

    def test_large_chain(self):
        chain = LinearChain(PitchShiftAugmentation(-1),
                            TimeStretchAugmentation(.95),
                            EqualizerAugmentation(300, 1, 2),
                            BackgroundNoiseAugmentation(.005),
                            TimeStretchAugmentation(1.25))
        augmented = chain(self.signal, self.sr)
        self.assertEqual(len(augmented), 1)


if __name__ == '__main__':
    unittest.main()
