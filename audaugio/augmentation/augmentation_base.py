class AugmentationBase:
    """
    Base class for an augmentation. Implement this to create your own augmentations.

    :param replaces: whether the augmentation should replace the audio it augments. Usually will be false.
    """

    def __init__(self, replaces: bool, *kwargs):
        self.replaces = replaces

    def augment(self, signal, sr):
        """
        Given a signal, apply the augmentation and return all resulting augmented audio as a list (even if it's a single signal).

        :param signal: unaugmented signal, ndarray
        :param sr: sample rate, int
        """
        raise NotImplementedError
