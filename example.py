import librosa

from audaugio.chain import CombinatoricChain
from audaugio.augmentation import BackgroundNoiseAugmentation, EqualizerAugmentation, PitchShiftAugmentation

y, sr = librosa.load('sample.wav')

chain = CombinatoricChain(PitchShiftAugmentation(1),
                          BackgroundNoiseAugmentation(.005),
                          EqualizerAugmentation(800, .15, -15))

augmented_audio = chain(y, sr)
for i, a in enumerate(augmented_audio):
    librosa.output.write_wav("output/{0}.wav".format(i), a, sr)
