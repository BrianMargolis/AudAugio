from audaugio import AugmentationBase
from .chain_base import ChainBase


class CombinatoricChain(ChainBase):
    """
    Apply augmentations combinatorically. This will return all possible combinations of non-replacing augmentations (i.e. when a non-replacing
    augmentation is performed, both the resulting augmented signal and the original signal are kept and augmented further). Note that some augmentations,
    like the windowing augmentation, always replace the audio they augment.

    :param args: an arbitrary amount of augmentations
    """

    def __init__(self, *augmentations: AugmentationBase):
        super().__init__(*augmentations)

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
