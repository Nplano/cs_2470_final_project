import librosa
# pip install librosa //this command to get this llibarar
import librosa.display

from matplotlib import pyplot as plt
import numpy as np


def int_to_str_4digts(val):

    string= str(val)
    while( len(string)<4):
        string = '0'+string

    return string

def main():
    print("my test")

    for i in range(1,10):
        path = '../data/500_songs/'+int_to_str_4digts(i)+'.wav'

        x, sr = librosa.load(path)
        hop_length = 512
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        S, phase = librosa.magphase(librosa.stft(x))
        rms = librosa.feature.rms(S=S)

        chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)

        oenv = librosa.onset.onset_strength(y= x, sr=sr, hop_length=hop_length)
        times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,hop_length=hop_length)# Estimate the global tempo for display purpose

        print("shape of spectrograph")
        print(np.shape(Xdb))

        print("shape of tempograph")
        print(np.shape(tempogram))

        print("shape of chromograph")
        print(np.shape(chromagram))

        print("shape of rms")
        print(np.shape(rms))

        #plt.plot(rms)
        plt.imshow(chromagram, interpolation='none')
        plt.show()













if __name__ == '__main__':
    main()
