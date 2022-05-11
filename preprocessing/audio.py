
import librosa
# pip install librosa //this command to get this llibarar
import librosa.display

#from matplotlib import pyplot as plt
import numpy as np

from numpy import asarray
from numpy import save

# load numpy array from npy file
from numpy import load

def int_to_str_4digts(val):

    string= str(val)
    while( len(string)<4):
        string = '0'+string

    return string






def get_audio_features(song_directory):
    #input- takes in the directory with all the song files in it
    # output, for all the songs returns a tuple of the following
    # 1.  spectrographs dimensions-> [number of songs,width, heigth]
    # 2.  tempographs  dimensions -> [number of songs,width, heigth]
    # 3.  chromographs dimensions -> [number of songs,width, heigth]
    # 4.  rms_energys  dimensions -> [number of songs,timestep]
    num_songs=2595
    hop_length = 512
    min=154624

    data=[]

    for i in range(1,num_songs+1):

        if(i !=1087 and i !=2174 and i !=2212 and
        i !=2240 and i !=2301 and i !=2359 and i !=2374 and i !=2491 and i !=2043):
            print(i)
            path = song_directory+'/'+int_to_str_4digts(i)+'.wav'
            x_t, sr = librosa.load(path)
            x_t = x_t[0:154624]




            S, phase = librosa.magphase(librosa.stft(x_t))
            rms = librosa.feature.rms(S=S)
            rms=np.reshape(rms,(np.shape(rms)[1]) )


            oenv = librosa.onset.onset_strength(y= x_t, sr=sr, hop_length=hop_length)
            times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)

            tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,hop_length=hop_length)[:,0:300]
            mfcc = librosa.feature.mfcc( y=x_t, sr=sr)[:,0:300]
            chromagram = librosa.feature.chroma_stft(y=x_t, sr=sr, hop_length=hop_length)[:,0:300]
            spectrograph = librosa.feature.melspectrogram(y=x_t,sr=sr)[:,0:300]
            spectral_contrast = librosa.feature.spectral_contrast( y=x_t, sr=sr)[:,0:300]
            tonnetz =librosa.feature.tonnetz( y=x_t, sr=sr)[:,0:300]

            tempogram_interp = np.zeros((60,1))
            mfcc_interp = np.zeros((60,1))
            chromagramm_interp = np.zeros((60,1))
            spectrograph_interp = np.zeros((60,1))
            spectral_contrast_interp = np.zeros((60,1))
            tonnetz_contrast_interp = np.zeros((60,1))


            for i in range(300):
                tempogram_col = tempogram[:, i]
                tempogram_col= np.reshape(np.interp(np.arange(60),np.arange(384),tempogram_col.reshape(-1)),(60,1))

                mfcc_col = mfcc[:, i]
                mfcc_col= np.reshape(np.interp(np.arange(60),np.arange(20),mfcc_col.reshape(-1)),(60,1))

                chromagram_col = chromagram[:, i]
                chromagram_col= np.reshape(np.interp(np.arange(60),np.arange(12),chromagram_col.reshape(-1)),(60,1))

                spectrograph_col = spectrograph[:, i]
                spectrograph_col= np.reshape(np.interp(np.arange(60),np.arange(128),spectrograph_col.reshape(-1)),(60,1))

                spectral_contrast_col = spectral_contrast[:, i]
                spectral_contrast_col=np.reshape( np.interp(np.arange(60),np.arange(7),spectral_contrast_col.reshape(-1)),(60,1))

                tonnetz_col = tonnetz[:, i]
                tonnetz_col= np.reshape(np.interp(np.arange(60),np.arange(6),tonnetz_col.reshape(-1)),(60,1))

                tempogram_interp = np.hstack((tempogram_interp, tempogram_col))
                mfcc_interp = np.hstack((mfcc_interp, mfcc_col))
                chromagramm_interp = np.hstack((chromagramm_interp, chromagram_col))
                spectrograph_interp = np.hstack((spectrograph_interp, spectrograph_col))
                spectral_contrast_interp = np.hstack((spectral_contrast_interp, spectral_contrast_col))
                tonnetz_contrast_interp = np.hstack((tonnetz_contrast_interp, tonnetz_col))

            tempogram_interp = tempogram_interp[:,1:]
            mfcc_interp = mfcc_interp[:,1:]
            chromagramm_interp = chromagramm_interp[:,1:]
            spectrograph_interp = spectrograph_interp[:,1:]
            spectral_contrast_interp = spectral_contrast_interp[:,1:]
            tonnetz_contrast_interp = tonnetz_contrast_interp[:,1:]
            out = np.concatenate((tempogram_interp,mfcc_interp,chromagramm_interp,spectrograph_interp,spectral_contrast_interp,tonnetz_contrast_interp),axis=-1)
            out = np.reshape(out,(60,300,6))
            data.append(out)

            print(np.shape(data))

    data = np.asarray(data)

    save('data.npy', data)


    pass


def audio_process_data(one_hot_labels,missing_indices):

    one_hot_labels = np.delete(one_hot_labels, missing_indices, axis=0)
    num_examples = np.shape(one_hot_labels)[0]
    num_training = int( 0.80*num_examples)
    one_hot_labels_training=one_hot_labels[0:num_training]
    one_hot_labels_testing =one_hot_labels[num_training:num_examples]
    data =  load('preprocessing/data.npy')
    data= np.delete(data, missing_indices, axis=0)

    data_training=data[0:num_training]
    data_testing =data[num_training:num_examples]

    return one_hot_labels_training,one_hot_labels_testing,data_training,data_testing



def main():
    get_audio_features('data/all_2595/')


if __name__ == '__main__':
    main()
