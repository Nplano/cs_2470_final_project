
import librosa
import librosa.display
import numpy as np
from numpy import save, load
import matplotlib.pyplot as plt


def index_to_filename(val):
    """
    Function to convert integer index to format of filename.
    :param val: integer index corresponding to filename
    :return filestring path
    """
    string = str(val)
    while len(string) < 4:
        string = '0'+string
    return f"{string}.wav"

def get_audio_features(song_directory):
    """
    Process each audio file and extract audio features to file data/numpy/audio_features.npy.
    :param song_directory: filepath directory location containing all audio clips (data/all_2595/)
    :return None
    """
    num_songs = 2595
    hop_length = 512
    sample_length = 154624
    data = []

    for i in range(1, num_songs+1):
        if i not in {1087, 2043, 2174, 2212, 2240, 2301, 2359, 2374, 2491}:
            path = f"{song_directory}/{index_to_filename(i)}"
            print(f"Processing file: {path}")
            x_t, sr = librosa.load(path)
            x_t = x_t[0:sample_length]

            S, phase = librosa.magphase(librosa.stft(x_t))
            rms = librosa.feature.rms(S=S)
            rms = np.reshape(rms,(np.shape(rms)[1]) )


            oenv = librosa.onset.onset_strength(y=x_t, sr=sr, hop_length=hop_length)
            times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)

            tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)[:, 0:300]
            mfcc = librosa.feature.mfcc(y=x_t, sr=sr)[:, 0:300]
            chromagram = librosa.feature.chroma_stft(y=x_t, sr=sr, hop_length=hop_length)[:, 0:300]
            spectrograph = librosa.feature.melspectrogram(y=x_t, sr=sr)[:, 0:300]
            spectral_contrast = librosa.feature.spectral_contrast(y=x_t, sr=sr)[:, 0:300]
            tonnetz = librosa.feature.tonnetz(y=x_t, sr=sr)[:, 0:300]

            tempogram_interp = np.zeros((60,1))
            mfcc_interp = np.zeros((60,1))
            chromagramm_interp = np.zeros((60,1))
            spectrograph_interp = np.zeros((60,1))
            spectral_contrast_interp = np.zeros((60,1))
            tonnetz_contrast_interp = np.zeros((60,1))





            for i in range(300):
                tempogram_col = tempogram[:, i]
                #tempogram_col = np.reshape(np.interp(np.arange(60),np.arange(384),tempogram_col.reshape(-1)),(60,1))
                tempogram_col = np.reshape(np.interp(np.linspace(0, 384, 60),np.arange(384),tempogram_col.reshape(-1)),(60,1))

                mfcc_col = mfcc[:, i]
                mfcc_col = np.reshape(np.interp(np.linspace(0, 20, 60),np.arange(20),mfcc_col.reshape(-1)),(60,1))


                chromagram_col = chromagram[:, i]
                chromagram_col = np.reshape(np.interp(np.linspace(0, 12, 60),np.arange(12),chromagram_col.reshape(-1)),(60,1))

                spectrograph_col = spectrograph[:, i]
                spectrograph_col = np.reshape(np.interp(np.linspace(0, 128, 60),np.arange(128),spectrograph_col.reshape(-1)),(60,1))

                spectral_contrast_col = spectral_contrast[:, i]
                spectral_contrast_col = np.reshape( np.interp(np.linspace(0, 7, 60),np.arange(7),spectral_contrast_col.reshape(-1)),(60,1))

                tonnetz_col = tonnetz[:, i]
                tonnetz_col = np.reshape(np.interp(np.linspace(0, 6, 60),np.arange(6),tonnetz_col.reshape(-1)),(60,1))

                tempogram_interp = np.hstack((tempogram_interp, tempogram_col))
                mfcc_interp = np.hstack((mfcc_interp, mfcc_col))
                chromagramm_interp = np.hstack((chromagramm_interp, chromagram_col))
                spectrograph_interp = np.hstack((spectrograph_interp, spectrograph_col))
                spectral_contrast_interp = np.hstack((spectral_contrast_interp, spectral_contrast_col))
                tonnetz_contrast_interp = np.hstack((tonnetz_contrast_interp, tonnetz_col))

            tempogram_interp = tempogram_interp[:, 1:]
            mfcc_interp = mfcc_interp[:, 1:]
            chromagramm_interp = chromagramm_interp[:, 1:]
            spectrograph_interp = spectrograph_interp[:, 1:]
            spectral_contrast_interp = spectral_contrast_interp[:, 1:]
            tonnetz_contrast_interp = tonnetz_contrast_interp[:, 1:]




            out = np.dstack((tempogram_interp,mfcc_interp,chromagramm_interp,spectrograph_interp,spectral_contrast_interp,tonnetz_contrast_interp))
            out = np.reshape(out, (60,300,6))

            data.append(out)

            print("Shape of data:", np.shape(data))

    data = np.asarray(data)
    save('data.npy', data)


def audio_process_data(one_hot_labels, missing_indices):
    """
    Given the mood labels and the indices to ignore, drop the data
    and labels which have corresponding audio files that are invalid.

    :param one_hot_labels: one hot encoded mood labels
    :param missing_indices: indices corresponding to bad audio files
    :return one_hot labels for training and testing sets,  data for training and testing sets
    """
    one_hot_labels = np.delete(one_hot_labels, missing_indices, axis=0)
    num_examples = np.shape(one_hot_labels)[0]
    num_training = int(0.80*num_examples)
    one_hot_labels_training = one_hot_labels[0:num_training]
    one_hot_labels_testing = one_hot_labels[num_training:num_examples]

    data = load('data/numpy/data.npy')
    data = np.delete(data, missing_indices, axis=0)
    data_training = data[0:num_training]
    data_testing = data[num_training:num_examples]

    return one_hot_labels_training, one_hot_labels_testing, data_training, data_testing


if __name__ == '__main__':
    get_audio_features('../all_2595/')
