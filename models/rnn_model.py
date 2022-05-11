import numpy as np
import tensorflow as tf
from models.our_loss import *

class RNN(tf.keras.Model):
    def __init__(self, vocab_size, max_sentence_size, with_loss):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_sentence_size = max_sentence_size

        self.word_embedding_size = 128
        self.hidden_state_size = 256
        self.epochs = 10

        self.with_loss = with_loss

        self.batch_size = 128
        self.learning_rate = 0.0005
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        self.embeddings = tf.Variable(tf.random.normal([self.vocab_size, self.word_embedding_size], stddev=.1))

        self.lstm = tf.keras.layers.LSTM(self.hidden_state_size, return_state=True, return_sequences=True)
        self.denseLayer_1 = tf.keras.layers.Dense(256, activation="relu")
        self.dropoutLayer_1 = tf.keras.layers.Dropout(0.2)
        self.denseLayer_2 = tf.keras.layers.Dense(128, activation="relu")
        self.dropoutLayer_2 = tf.keras.layers.Dropout(0.2)
        self.denseLayer_3 = tf.keras.layers.Dense(64, activation="relu")
        self.dropoutLayer_3 = tf.keras.layers.Dropout(0.2)

        self.vec2veclayer = tf.keras.layers.Dense(2, activation="tanh")
        self.softmax_layer = tf.keras.layers.Dense(4, activation="softmax")




    @tf.function
    def call(self, inputs):
        embeddings = tf.nn.embedding_lookup(self.embeddings, inputs)
        whole_seq_output, final_memory_state, final_carry_state = self.lstm(embeddings, initial_state=None)
        output = self.denseLayer_1(final_memory_state)
        output = self.dropoutLayer_1(output)
        output = self.denseLayer_2(output)
        output = self.dropoutLayer_2(output)
        output = self.denseLayer_3(output)
        output = self.dropoutLayer_3(output)
        probs = self.softmax_layer(output)
        vec2vec = self.vec2veclayer(output)

        return probs,vec2vec

    def accuracy_function(self, probs, labels):
        predictions = tf.argmax(input=probs, axis=1)
        labels = tf.argmax(input=labels, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        return accuracy

    def loss_function(self, probs, labels,two_d_vectorspace):
        cce = tf.keras.losses.CategoricalCrossentropy()

        C = our_loss( labels,two_d_vectorspace, self.batch_size)
        loss = cce(labels, probs)
        if(self.with_loss==True):
            loss = loss +C
        return loss


def train_rnn(model, inputs, labels, test_inputs,test_labels):
    train_acc =[]
    test_acc =[]
    epochd = []
    for epoch in range(model.epochs):
        print("epoch = ", epoch)
        indices = np.arange(len(inputs))
        indices = tf.random.shuffle(indices)
        inputs_shuffled = tf.gather(inputs, indices)
        labels_shuffled = tf.gather(labels, indices)

        batch_size = model.batch_size
        batch_num = inputs.shape[0] // batch_size
        for i in range(batch_num):
            train_batch = inputs_shuffled[i * batch_size: (i + 1) * batch_size]
            labels_batch = labels_shuffled[i * batch_size: (i + 1) * batch_size]
            with tf.GradientTape() as tape:
                probs,vec2vec = model.call(train_batch)
                loss = model.loss_function(probs, labels_batch,vec2vec)
                print("loss = ", loss.numpy())

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_acc.append(test_rnn(model, inputs, labels))
        test_acc.append(test_rnn(model, test_inputs, test_labels))
        epochd.append(epoch)

    return train_acc,test_acc,epochd

def test_rnn(model, inputs, labels):
    batch_size = model.batch_size
    batch_num = inputs.shape[0] // batch_size
    acc_sum = 0
    for i in range(batch_num):
        test_batch = inputs[i * batch_size: (i + 1) * batch_size]
        labels_batch = labels[i * batch_size: (i + 1) * batch_size]
        probs_batch, vec2vex = model.call(test_batch)
        acc_sum += model.accuracy_function(probs_batch, labels_batch)

    return acc_sum / batch_num
