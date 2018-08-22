import librosa

import audaugio

y, sr = librosa.load('sample.wav')

chain = audaugio.CombinatoricChain(audaugio.PitchShiftAugmentation(1),
                                   audaugio.BackgroundNoiseAugmentation(.005),
                                   audaugio.EqualizerAugmentation(800, .15, -15))

augmented_audio = chain(y, sr)

for i, a in enumerate(augmented_audio):
    librosa.output.write_wav("output/{0}.wav".format(i), a, sr)
