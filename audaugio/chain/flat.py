from audaugio import AugmentationBase
from .chain_base import ChainBase


class FlatChain(ChainBase):
    """
    Apply augmentations without layering any on top of others. The signal is modified by each augmentation and saved. Useful for qualitatively analyzing
    augmentations.

    :param augmentations: an arbitrary amount of augmentations
    """

    def __init__(self, *augmentations: AugmentationBase):
        super().__init__(*augmentations)

    def _apply_augmentations(self, signal: [], sr: int):
        augmented_audio = []
        for augmentation in self._augmentations:
            augmented_audio += augmentation.augment(signal, sr)

        return augmented_audio
