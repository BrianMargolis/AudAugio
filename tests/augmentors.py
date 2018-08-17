import unittest
from typing import List

import numpy as np

from audiaug.augmentors.background_noise import BackgroundNoiseAugmentation
from audiaug.augmentors.time_stretch import TimeStretchAugmentation
from audiaug.augmentors.windowing import WindowingAugmentation


class TestAugmentor(unittest.TestCase):
    def setUp(self):
        # mock a signal
        self.len_sec = 8
        self.sr = 16000
        self.mock_audio = np.random.random(size=self.sr * self.len_sec)


class TestWindowing(TestAugmentor):
    def setUp(self):
        super().setUp()
        self.window_length = 4
        self.hop_size = 2
        self.augmentor = WindowingAugmentation(self.window_length, self.hop_size)

    def test_number_of_segments(self):
        expected_n_segments = np.ceil(self.len_sec / (self.window_length / self.hop_size)) - 1
        augmented = self.augmentor.augment(self.mock_audio, self.sr)
        self.assertEqual(expected_n_segments, len(augmented))

    def test_returns_array(self):
        augmented = self.augmentor.augment(self.mock_audio, self.sr)
        self.assertIsInstance(augmented, List)


class TestBackgroundNoise(TestAugmentor):
    def setUp(self):
        super().setUp()
        self.amplitude = .005
        self.augmentor = BackgroundNoiseAugmentation(self.amplitude)

    def test_length(self):
        augmented = self.augmentor.augment(self.mock_audio, self.sr)
        self.assertEqual(len(augmented[0]), len(self.mock_audio))

    def test_returns_array(self):
        augmented = self.augmentor.augment(self.mock_audio, self.sr)
        self.assertIsInstance(augmented, List)


class TestTimeStretch(TestAugmentor):
    def test_speed_up(self):
        augmentor = TimeStretchAugmentation(1 + .1 * np.random.random())
        augmented = augmentor.augment(self.mock_audio, self.sr)
        self.assertLess(len(augmented[0]), len(self.mock_audio))

    def test_slow_down(self):
        augmentor = TimeStretchAugmentation(1 - .1 * np.random.random())
        augmented = augmentor.augment(self.mock_audio, self.sr)
        self.assertGreater(len(augmented[0]), len(self.mock_audio))

    def test_no_change(self):
        augmentor = TimeStretchAugmentation(1)
        augmented = augmentor.augment(self.mock_audio, self.sr)
        self.assertEqual(len(augmented[0]), len(self.mock_audio))

    def test_returns_array(self):
        augmentor = TimeStretchAugmentation(1.05)
        augmented = augmentor.augment(self.mock_audio, self.sr)
        self.assertIsInstance(augmented, List)


if __name__ == '__main__':
    unittest.main()
