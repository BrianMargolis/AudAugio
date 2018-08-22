import unittest
from typing import List

import numpy as np

from audaugio.augmentation.windowing import WindowingAugmentation
from audaugio.augmentation.equalizer import EqualizerAugmentation
from audaugio.augmentation.time_stretch import TimeStretchAugmentation
from audaugio.augmentation.pitch_shift import PitchShiftAugmentation
from audaugio.augmentation.background import BackgroundNoiseAugmentation


class TestAugmentor(unittest.TestCase):
    def setUp(self):
        # mock a random signal
        self.len_sec = np.random.randint(low=5, high=20)
        self.sr = np.random.randint(low=16000, high=44000)
        self.audio = self.mock_audio(self.len_sec, self.sr)

    @staticmethod
    def mock_audio(len_sec, sr):
        return np.random.random(size=sr * len_sec)


class TestWindowing(TestAugmentor):
    @staticmethod
    def expected_segments(len_sec, window_length, hop_size, drop_last):
        if window_length >= len_sec:
            return 1

        if not drop_last:
            padded_len = len_sec + (len_sec - window_length) % hop_size
        else:
            padded_len = len_sec

        n_segments = np.floor((padded_len - window_length) / hop_size + 1)
        return n_segments

    def test_number_of_segments_evenly_divisible(self):
        window_length = 4
        hop_size = 2

        len_sec = np.arange(start=1, stop=100)
        for ls in len_sec:
            sr = 1
            self._test_number_of_segments(hop_size, ls, sr, window_length, False)

        len_sec = np.arange(start=1, stop=100)
        for ls in len_sec:
            sr = 1
            self._test_number_of_segments(hop_size, ls, sr, window_length, True)

    def test_number_of_segments_non_evenly_divisible(self):
        window_length = 5
        hop_size = 2

        len_sec = np.arange(start=1, stop=100)
        for ls in len_sec:
            sr = 1
            self._test_number_of_segments(hop_size, ls, sr, window_length, False)

        len_sec = np.arange(start=1, stop=100)
        for ls in len_sec:
            sr = 1
            self._test_number_of_segments(hop_size, ls, sr, window_length, True)

    def _test_number_of_segments(self, hop_size, len_sec, sr, window_length, drop_last):
        augmentor = WindowingAugmentation(window_length, hop_size, drop_last=drop_last)
        audio = self.mock_audio(len_sec, sr)
        expected_n_segments = self.expected_segments(len_sec, window_length, hop_size, drop_last)
        augmented = augmentor.augment(audio, sr)
        # print("{0}:\t{1}\t{2}".format(len_sec, int(expected_n_segments), len(augmented)))
        self.assertEqual(expected_n_segments, len(augmented))

    def test_returns_array(self):
        window_length = 5
        hop_size = 2
        augmentor = WindowingAugmentation(window_length, hop_size)

        augmented = augmentor.augment(self.audio, self.sr)
        self.assertIsInstance(augmented, List)


class TestBackgroundNoise(TestAugmentor):
    def setUp(self):
        super().setUp()
        self.amplitude = .005
        self.augmentor = BackgroundNoiseAugmentation(self.amplitude)

    def test_length(self):
        augmented = self.augmentor.augment(self.audio, self.sr)
        self.assertEqual(len(augmented[0]), len(self.audio))

    def test_returns_array(self):
        augmented = self.augmentor.augment(self.audio, self.sr)
        self.assertIsInstance(augmented, List)


class TestTimeStretch(TestAugmentor):
    def test_speed_up(self):
        augmentor = TimeStretchAugmentation(1 + .1 * np.random.random())
        augmented = augmentor.augment(self.audio, self.sr)
        self.assertLess(len(augmented[0]), len(self.audio))

    def test_slow_down(self):
        augmentor = TimeStretchAugmentation(1 - .1 * np.random.random())
        augmented = augmentor.augment(self.audio, self.sr)
        self.assertGreater(len(augmented[0]), len(self.audio))

    def test_returns_array(self):
        augmentor = TimeStretchAugmentation(1.05)
        augmented = augmentor.augment(self.audio, self.sr)
        self.assertIsInstance(augmented, List)


class TestPitchShift(TestAugmentor):
    def test_pitch_up(self):
        steps = np.random.randint(low=1, high=5)
        augmentor = PitchShiftAugmentation(steps)
        augmented = augmentor.augment(self.audio, self.sr)
        self.assertEqual(len(augmented), 1)
        self.assertEqual(len(augmented[0]), len(self.audio))

    def test_pitch_down(self):
        steps = -np.random.randint(low=1, high=5)
        augmentor = PitchShiftAugmentation(steps)
        augmented = augmentor.augment(self.audio, self.sr)
        self.assertEqual(len(augmented), 1)
        self.assertEqual(len(augmented[0]), len(self.audio))

    def test_returns_array(self):
        augmentor = PitchShiftAugmentation(1)
        augmented = augmentor.augment(self.audio, self.sr)
        self.assertIsInstance(augmented, List)


class TestEqualizer(TestAugmentor):
    def test_low_pass(self):
        freq = 16000
        q = .1
        gain = 1
        augmentor = EqualizerAugmentation(freq, q, gain)
        augmented = augmentor.augment(self.audio, self.sr)
        self.assertEqual(len(augmented), 1)
        self.assertEqual(len(augmented[0]), len(self.audio))

    def test_high_pass(self):
        freq = 400
        q = .2
        gain = 3
        augmentor = EqualizerAugmentation(freq, q, gain)
        augmented = augmentor.augment(self.audio, self.sr)
        self.assertEqual(len(augmented), 1)
        self.assertEqual(len(augmented[0]), len(self.audio))

    def test_returns_array(self):
        freq = 800
        q = .15
        gain = 1
        augmentor = EqualizerAugmentation(freq, q, gain)
        augmented = augmentor.augment(self.audio, self.sr)
        self.assertIsInstance(augmented, List)


if __name__ == '__main__':
    unittest.main()
