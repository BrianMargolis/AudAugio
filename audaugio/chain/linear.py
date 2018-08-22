from audaugio import AugmentationBase
from .chain_base import ChainBase


class LinearChain(ChainBase):
    """
    Apply augmentations linearly. The signal is modified by each augmentation in the order that they are passed into the constructor.

    :param args: an arbitrary amount of augmentations
    """

    def __init__(self, *augmentations: AugmentationBase):
        super().__init__(*augmentations)

    def _apply_augmentations(self, signal: [], sr: int):
        augmented_audio = [signal]
        for augmentation in self._augmentations:
            augmented_audio_batch = []
            for signal in augmented_audio:
                augmented_audio_batch += augmentation.augment(signal, sr)

            augmented_audio = augmented_audio_batch

        return augmented_audio
