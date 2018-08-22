import numpy as np

from audaugio import AugmentationBase


class ChainBase:
    def __init__(self, *args: AugmentationBase):
        self._augmentations = list(args)

    def __add__(self, new_augmentation: AugmentationBase):
        self._augmentations.append(new_augmentation)

    def __call__(self, audio: np.ndarray, sr: int):
        return self._apply_augmentations(audio, sr)

    def _apply_augmentations(self, audio, sr):
        raise NotImplementedError