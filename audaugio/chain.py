import numpy as np

from audaugio.augmentation import Augmentation


class AugmentationChain:
    def __init__(self, *args: Augmentation):
        self._augmentations = list(args)

    def __add__(self, new_augmentation: Augmentation):
        self._augmentations.append(new_augmentation)

    def __call__(self, audio: np.ndarray, sr: int):
        return self._apply_augmentations(audio, sr)

    def _apply_augmentations(self, audio, sr):
        raise NotImplementedError


class CombinatoricChain(AugmentationChain):
    def __init__(self, *args: Augmentation):
        """
        Apply augmentations combinatorically. This will return all possible combinations of non-replacing augmentations (i.e. when a non-replacing
        augmentation is performed, both the resulting augmented signal and the original signal are kept and augmented further). Note that some augmentations,
        like the windowing augmentation, always replace the audio they augment.

        :param args: an arbitrary amount of augmentations
        """
        super().__init__(*args)

    def _apply_augmentations(self, signal: [], sr: int):
        # start with just the original audio and then apply all augmentations
        augmented_audio = [signal]
        for augmentation in self._augmentations:
            augmented_audio_batch = []
            for signal in augmented_audio:
                augmented_audio_batch += augmentation.augment(signal, sr)

            if augmentation.replaces:  # e.g. windowing augmentation, which replaces the original audio with windowed versions
                augmented_audio = augmented_audio_batch
            else:  # e.g. time stretching augmentation
                augmented_audio += augmented_audio_batch

        return augmented_audio


class LinearChain(AugmentationChain):
    def __init__(self, *args: Augmentation):
        """
        Apply augmentations linearly. The signal is modified by each augmentation in the order that they are passed into the constructor.

        :param args: an arbitrary amount of augmentations
        """
        super().__init__(*args)

    def _apply_augmentations(self, signal: [], sr: int):
        augmented_audio = [signal]
        for augmentation in self._augmentations:
            augmented_audio_batch = []
            for signal in augmented_audio:
                augmented_audio_batch += augmentation.augment(signal, sr)

            augmented_audio = augmented_audio_batch

        return augmented_audio
