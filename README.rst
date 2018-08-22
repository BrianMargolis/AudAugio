AudAugio
===========

AudAugio (pronounced like “adagio”) is a Python library for augmenting
audio for machine learning. It includes a built-in set of common
augmentations as well as the ability to easily define new ones, as well
as a framework for applying and layering those augmentations.

Installation
------------

Install AudAugio with pip:

::

   pip install audaugio

If you want to use one of the built-in augmentations that relies on SoX,
you’ll need to install it separately. You can do that at `the SoX
SourceForge page`_. Augmentations that rely on SoX note that in their
documentation, and are listed here:

- Equalizer Augmentation
- Low Pass Augmentation
- High Pass Augmentation

Use
---

Below is an example of how to use AudAugio to augment an audio file:

.. code:: python

    import librosa

    import audaugio

    y, sr = librosa.load('sample.wav')

    chain = audaugio.CombinatoricChain(audaugio.PitchShiftAugmentation(1),
                                       audaugio.BackgroundNoiseAugmentation(.005),
                                       audaugio.EqualizerAugmentation(800, .15, -15))

    augmented_audio = chain(y, sr)

    for i, a in enumerate(augmented_audio):
        librosa.output.write_wav("output/{0}.wav".format(i), a, sr)

For more, read `the documentation for AudAugio`_.


Augmentation Chains
-------------------

Augmentations are applied through augmentation chains. There are two
kinds of chains - combinatoric chains and linear chains.
Combinatorically applying the augmentations creates a modified and
unmodified version of the signal for each augmentation, which are then
each augmented further by the remaining augmentations in the chain. For
example, if a single signal is combinatorically augmented with both a
pitch shift augmentation and a background noise augmentation, there will
be four resulting augmented signals:

- The dry signal
- The pitch shifted signal
- The signal with background noise added
- The pitch shifted signal with background noise added

Linear chains do not retain the unmodified signal after processing. If
same chain as above was applied, the only signal returned would be the
pitch shifted signal with background noise added.

Contact
-------

Contact Brian Margolis (brianmargolis [at] u.northwestern.edu) with any
questions or issues. Please look at the “issues” page before reporting
problems.

.. _the SoX SourceForge page: https://sourceforge.net/projects/sox/files/sox/
.. _the documentation for AudAugio: https://audaugio.readthedocs.io/