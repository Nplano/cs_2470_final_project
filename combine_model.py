import numpy as np
import tensorflow as tf
from audio_model import audio_model
from rnn_model import *

class Combine_model(tf.keras.Model):
    def __init__(self, rnn_model, audio_model):
        super(Combine_model, self).__init__()

        self.rnn_model =rnn_model
        self.audio_model = audio_model

        self.epochs = 30

        self.batch_size = 128
        self.learning_rate = 0.0005
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        self.denseLayer_1 = tf.keras.layers.Dense(256, activation="relu")
        self.dropoutLayer_1 = tf.keras.layers.Dropout(0.2)
        self.denseLayer_2 = tf.keras.layers.Dense(128, activation="relu")
        self.dropoutLayer_2 = tf.keras.layers.Dropout(0.2)
        self.denseLayer_3 = tf.keras.layers.Dense(64, activation="relu")
        self.dropoutLayer_3 = tf.keras.layers.Dropout(0.2)
        self.denseLayer_4 = tf.keras.layers.Dense(4, activation="softmax")




    @tf.function
    def call(self, probs_rnn, probs_audio):
        print(np.shape(probs_rnn))
        print(np.shape(probs_audio))

        combined = tf.concat([probs_rnn,probs_audio],1)
        output = self.denseLayer_1(combined)
        output = self.dropoutLayer_1(output)
        output = self.denseLayer_2(output)
        output = self.dropoutLayer_2(output)
        output = self.denseLayer_3(output)
        output = self.dropoutLayer_3(output)
        probs = self.denseLayer_4(output)
        return probs

    def accuracy_function(self, probs, labels):
        predictions = tf.argmax(input=probs, axis=1)
        labels = tf.argmax(input=labels, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        return accuracy

    def loss_function(self, probs, labels):
        cce = tf.keras.losses.CategoricalCrossentropy()
        loss = cce(labels, probs)
        return loss


def train_combine(model,  rnn_inputs, audio_inputs,labels):

    for epoch in range(model.epochs):
        print("epoch = ", epoch)

        indices = np.arange(len(rnn_inputs))
        indices = tf.random.shuffle(indices)
        rnn_inputs_shuffled = tf.gather(rnn_inputs, indices)
        labels_shuffled = tf.gather(labels, indices)
        shuffled_audio_inputs=  tf.gather(audio_inputs,indices)

        batch_size = model.batch_size
        batch_num = rnn_inputs.shape[0] // batch_size
        for i in range(batch_num):
            rnn_train_batch = rnn_inputs_shuffled[i * batch_size: (i + 1) * batch_size]
            labels_batch = labels_shuffled[i * batch_size: (i + 1) * batch_size]
            audio_inputs_train_batch = shuffled_audio_inputs[i * batch_size: (i + 1) * batch_size]

            probs_rnn =model.rnn_model.call(rnn_train_batch)
            probs_audio =model.audio_model.call(audio_inputs_train_batch )
            with tf.GradientTape() as tape:
                probs = model.call(probs_rnn,probs_audio)
                loss = model.loss_function(probs, labels_batch)
                print("loss = ", loss.numpy())

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    pass

    #return test_rnn(model,  rnn_inputs, labels,spectrograph,chromograph,tempograph,rms_energy)

def test_combine(model,  rnn_inputs, audio_inputs,labels):
    batch_size = model.batch_size
    batch_num = inputs.shape[0] // batch_size
    acc_sum = 0


    for i in range(batch_num):
        rnn_inputs_test = rnn_inputs[i * batch_size: (i + 1) * batch_size]
        labels_batch = labels[i * batch_size: (i + 1) * batch_size]
        audio_inputs_test = audio_inputs[i * batch_size: (i + 1) * batch_size]
        probs_rnn =model.rnn_model.call(rnn_inputs_test)
        probs_audio =model.audio_model.call( spectrograph_test, tempograph_test,chromograph_test,rms_trarms_energy_testin_batch)
        probs_batch = model.call(probs_rnn,probs_audio)
        acc_sum += model.accuracy_function(probs_batch, labels_batch)

    return acc_sum / batch_num
