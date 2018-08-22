import librosa

from audaugio import EqualizerAugmentation, BackgroundNoiseAugmentation, PitchShiftAugmentation
from audaugio.chain.combinatoric import CombinatoricChain

y, sr = librosa.load('sample.wav')

chain = CombinatoricChain(PitchShiftAugmentation(1),
                          BackgroundNoiseAugmentation(.005),
                          EqualizerAugmentation(800, .15, -15))

augmented_audio = chain(y, sr)
for i, a in enumerate(augmented_audio):
    librosa.output.write_wav("output/{0}.wav".format(i), a, sr)
