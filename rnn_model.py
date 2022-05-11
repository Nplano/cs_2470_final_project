import numpy as np
import tensorflow as tf

class RNN(tf.keras.Model):
    def __init__(self, vocab_size, max_sentence_size):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_sentence_size = max_sentence_size

        self.word_embedding_size = 128
        self.hidden_state_size = 256
        self.epochs = 1

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
        self.denseLayer_4 = tf.keras.layers.Dense(4, activation="softmax")


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


def train_rnn(model, inputs, labels):
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
                probs = model.call(train_batch)
                loss = model.loss_function(probs, labels_batch)
                print("loss = ", loss.numpy())

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return test_rnn(model, inputs, labels)

def test_rnn(model, inputs, labels):
    batch_size = model.batch_size
    batch_num = inputs.shape[0] // batch_size
    acc_sum = 0
    for i in range(batch_num):
        test_batch = inputs[i * batch_size: (i + 1) * batch_size]
        labels_batch = labels[i * batch_size: (i + 1) * batch_size]
        probs_batch = model.call(test_batch)
        acc_sum += model.accuracy_function(probs_batch, labels_batch)

    return acc_sum / batch_num
