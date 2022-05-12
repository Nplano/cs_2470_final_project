import os
import numpy as np
import tensorflow as tf
from numpy import load
from preprocessing.labels import get_labels
from preprocessing.lyrics import *
from preprocessing.audio import *
from models.audio_model import AudioModel
from models.rnn_model import *
from models.combined_model import *
import librosa
import librosa.display
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    # uncomment to run the two sub-models
    #run_audio_model()
    #print(f"MAX_SENTENCE_SIZE={MAX_SENTENCE_SIZE}")
    train_acc,test_acc,epoch=run_lyrics_model(False)
    fig, ax = plt.subplots()
    plt.title("Train and Test accuracy for RNN model without DFB")
    plt.xlabel('Epoch')
    ax.plot(train_acc, color = 'blue', label = 'Train accuracy ')
    ax.plot(test_acc, color = 'red', label = 'Test accuracy')
    ax.legend(loc = 'upper left')
    plt.show()

    train_acc,test_acc,epoch=run_lyrics_model(True)
    fig, ax = plt.subplots()
    plt.title("Train and Test accuracy for RNN model with DFB loss")
    plt.xlabel('Epoch')
    ax.plot(train_acc, color = 'blue', label = 'Train accuracy')
    ax.plot(test_acc, color = 'red', label = 'Test accuracy')
    ax.legend(loc = 'upper left')
    plt.show()
    # #

    train_acc,test_acc,epoch=run_audio_model(False,False)
    fig, ax = plt.subplots()
    plt.title("Train and Test accuracy for Audio model without DFB loss")
    plt.xlabel('Epoch')
    ax.plot(train_acc, color = 'blue', label = 'Train accuracy')
    ax.plot(test_acc, color = 'red', label = 'Test accuracy')
    ax.legend(loc = 'upper left')
    plt.show()


    fig, ax = plt.subplots()
    plt.title("Train and Test accuracy for Audio model with Tempo Feature")
    plt.xlabel('Epoch')
    ax.plot(train_acc, color = 'blue', label = 'Train accuracy')
    ax.plot(test_acc, color = 'red', label = 'Test accuracy')
    ax.legend(loc = 'upper left')
    plt.show()


    train_acc,test_acc,epoch=run_audio_model(True,False)
    fig, ax = plt.subplots()
    plt.title("Train and Test accuracy for Audio model with DFB loss")
    plt.xlabel('Epoch')
    ax.plot(train_acc, color = 'blue', label = 'Train accuracy')
    ax.plot(test_acc, color = 'red', label = 'Test accuracy')
    ax.legend(loc = 'upper left')
    plt.show()

    train_acc,test_acc,epoch=run_audio_model(True,True)
    fig, ax = plt.subplots()
    plt.title("Train and Test accuracy for Audio model without Tempo Feature")
    plt.xlabel('Epoch')
    ax.plot(train_acc, color = 'blue', label = 'Train accuracy')
    ax.plot(test_acc, color = 'red', label = 'Test accuracy')
    ax.legend(loc = 'upper left')
    plt.show()

    #
    train_acc,test_acc,epoch=run_combined()
    fig, ax = plt.subplots()
    plt.title("Train and Test accuracy for combined model")
    plt.xlabel('Epoch')
    ax.plot(train_acc, color = 'blue', label = 'Train accuracy ')
    ax.plot(test_acc, color = 'red', label = 'Test accuracy')
    ax.legend(loc = 'upper left')
    plt.show()


    # training_lyrics_ids, testing_lyrics_ids, vocab, lyric_missing_indices, pad_token_ind = get_lyric_data("data/lyrics.txt")
    # one_hot_labels = get_labels('data/moods.txt')
    # training_labels, testing_labels = lyric_labels_process(one_hot_labels, lyric_missing_indices)
    # spectrographs =  load('preprocessing/spectrographs.npy')
    # spectrographs = np.delete(spectrographs, lyric_missing_indices, axis=0)
    # print(np.shape(training_labels))
    # print(np.shape(testing_labels))
    #
    # print(np.shape(spectrographs))

def run_audio_model(with_loss, without_tempo):

    one_hot_labels = get_labels('data/moods.txt')
    training_lyrics_ids, testing_lyrics_ids, vocab, lyric_missing_indices, pad_token_ind = get_lyric_data("data/lyrics.txt")
    one_hot_labels_training,one_hot_labels_testing, data_train,data_test= audio_process_data(one_hot_labels,lyric_missing_indices)




    model = AudioModel(with_loss,without_tempo)

    return model.train(one_hot_labels_training,data_train,data_test,one_hot_labels_testing)



def run_lyrics_model(with_loss):
    training_lyrics_ids, testing_lyrics_ids, vocab, lyric_missing_indices, pad_token_ind = get_lyric_data("data/lyrics.txt")
    one_hot_labels = get_labels('data/moods.txt')
    model = RNN(len(vocab), MAX_SENTENCE_SIZE,with_loss)
    training_labels, testing_labels = lyric_labels_process(one_hot_labels, lyric_missing_indices)
    return train_rnn(model, training_lyrics_ids, training_labels,testing_lyrics_ids,testing_labels)





def run_combined():
    training_lyrics_ids, testing_lyrics_ids, vocab, lyric_missing_indices, pad_token_ind = get_lyric_data("data/lyrics.txt")
    one_hot_labels = get_labels('data/moods.txt')
    rnn_model = RNN(len(vocab), MAX_SENTENCE_SIZE,False)
    training_labels, testing_labels = lyric_labels_process(one_hot_labels, lyric_missing_indices)


    one_hot_labels_training,one_hot_labels_testing, data_train,data_test = audio_process_data(one_hot_labels, lyric_missing_indices)
    model_audio = AudioModel(False,False)
    train_acc,test_acc,epoch = model_audio.train(one_hot_labels_training,data_train,data_test,one_hot_labels_testing)

    fig, ax = plt.subplots()
    plt.title("Train and Test accuracy for the Audio Model of the combined model")
    plt.xlabel('Epoch')
    ax.plot(train_acc, color = 'blue', label = 'Train accuracy ')
    ax.plot(test_acc, color = 'red', label = 'Test accuracy')
    ax.legend(loc = 'upper left')
    plt.show()


    train_acc,test_acc,epoch=train_rnn(rnn_model, training_lyrics_ids, training_labels,testing_lyrics_ids,testing_labels)

    fig, ax = plt.subplots()
    plt.title("Train and Test accuracy for the RNN of the combined model")
    plt.xlabel('Epoch')
    ax.plot(train_acc, color = 'blue', label = 'Train accuracy ')
    ax.plot(test_acc, color = 'red', label = 'Test accuracy')
    ax.legend(loc = 'upper left')
    plt.show()



    combined_model = CombinedModel(rnn_model, model_audio)

    return train_combine(combined_model, training_lyrics_ids,data_train,one_hot_labels_training, testing_lyrics_ids,data_test,one_hot_labels_testing )



if __name__ == '__main__':
    main()
