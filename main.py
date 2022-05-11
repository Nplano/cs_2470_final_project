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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    # uncomment to run the two sub-models
    run_audio_model()
    #print(f"MAX_SENTENCE_SIZE={MAX_SENTENCE_SIZE}")
    #run_lyrics_model()
    #run_combined()


    # training_lyrics_ids, testing_lyrics_ids, vocab, lyric_missing_indices, pad_token_ind = get_lyric_data("data/lyrics.txt")
    # one_hot_labels = get_labels('data/moods.txt')
    # training_labels, testing_labels = lyric_labels_process(one_hot_labels, lyric_missing_indices)
    # spectrographs =  load('preprocessing/spectrographs.npy')
    # spectrographs = np.delete(spectrographs, lyric_missing_indices, axis=0)
    # print(np.shape(training_labels))
    # print(np.shape(testing_labels))
    #
    # print(np.shape(spectrographs))

def run_audio_model():
    #call  get_audio_features
    #call get get_lyrics
    #call get_labels will return one hot encoding as well as 2d vecotor space represetnation
    one_hot_labels = get_labels('data/moods.txt')
    training_lyrics_ids, testing_lyrics_ids, vocab, lyric_missing_indices, pad_token_ind = get_lyric_data("data/lyrics.txt")
    one_hot_labels_training,one_hot_labels_testing, data_train,data_test= audio_process_data(one_hot_labels,lyric_missing_indices)
    model = AudioModel()
    model.train(one_hot_labels_training,data_train)

    acc= model.test(one_hot_labels_testing,data_test)

    print("test accuracy ")
    print(acc)

    acc_train = model.test( one_hot_labels_training,data_train)
    print("train accuracy ")
    print(acc_train)


def run_lyrics_model():
    training_lyrics_ids, testing_lyrics_ids, vocab, lyric_missing_indices, pad_token_ind = get_lyric_data("data/lyrics.txt")
    one_hot_labels = get_labels('data/moods.txt')
    model = RNN(len(vocab), MAX_SENTENCE_SIZE)

    # remove labels for missing lyrics
    training_labels, testing_labels = lyric_labels_process(one_hot_labels, lyric_missing_indices)

    acc_train = train_rnn(model, training_lyrics_ids, training_labels)
    print("train acc", acc_train)
    acc_test = test_rnn(model, testing_lyrics_ids, testing_labels)
    print("test acc", acc_test)


def run_combined():
    training_lyrics_ids, testing_lyrics_ids, vocab, lyric_missing_indices, pad_token_ind = get_lyric_data("data/lyrics.txt")
    one_hot_labels = get_labels('data/moods.txt')
    rnn_model = RNN(len(vocab), MAX_SENTENCE_SIZE)
    training_labels, testing_labels = lyric_labels_process(one_hot_labels, lyric_missing_indices)

    #train rnn
    train_rnn(rnn_model, training_lyrics_ids, training_labels)

    one_hot_labels = get_labels('data/moods.txt')
    one_hot_labels_training,one_hot_labels_testing, data_train,data_test = audio_process_data(one_hot_labels, lyric_missing_indices)
    model_audio = AudioModel()
    #train audio
    model_audio.train(one_hot_labels_training, data_train)

    combined_model = CombinedModel(rnn_model, model_audio)

    #train combined model
    train_combine(combined_model, training_lyrics_ids,data_train,one_hot_labels_training )
    acc = test_combine(combined_model, testing_lyrics_ids,data_test,one_hot_labels_testing )
    print("combined model acc")
    print(acc)

if __name__ == '__main__':
    main()
