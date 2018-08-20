from typing import List

from audaugio.augmentors.generics import Augmentation


def apply_augmentations(augmentations: List[Augmentation], signal: [], sr: int):
    # start with just the original audio and then apply all augmentations
    augmented_audio = [signal]
    for augmentation in augmentations:
        augmented_audio_batch = []
        for signal in augmented_audio:
            augmented_audio_batch += augmentation.augment(signal, sr)

        if augmentation.replaces:  # e.g. windowing augmentation, which replaces the original audio with windowed versions
            augmented_audio = augmented_audio_batch
        else:  # e.g. time stretching augmentation
            augmented_audio.append(augmented_audio_batch)

    return augmented_audio
