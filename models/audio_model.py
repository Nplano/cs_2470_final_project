import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from models.our_loss import *


class AudioModel(tf.keras.Model):
    def __init__(self, with_loss, without_tempo):
        super(AudioModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.batch_size = 128
        self.num_epochs = 30
        self.num_classes = 4
        self.with_loss = with_loss
        self.without_tempo =without_tempo
        input_shape = (60,300,6)
        if(without_tempo==True):
            input_shape = (60,300,5)

        self.cnn = Sequential()

        self.cnn.add(tf.keras.layers.Conv2D(20, kernel_size =(6, 6),input_shape=input_shape ,kernel_regularizer=tf.keras.regularizers.L2(l2=0.0000001)))
        #self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.ReLU())
        self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.cnn.add(tf.keras.layers.Conv2D(40, kernel_size =(5, 5), padding ='same',kernel_regularizer=tf.keras.regularizers.L2(l2=0.0000001)))
    #    self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.ReLU())
        self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.cnn.add(tf.keras.layers.Conv2D(80, kernel_size =(4, 4), padding ='same',kernel_regularizer=tf.keras.regularizers.L2(l2=0.0000001)))
        self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.ReLU())
        self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))



        self.cnn.add(Flatten())
        self.cnn.add(tf.keras.layers.Dense(64 ,kernel_regularizer=tf.keras.regularizers.L2(l2=0.0000001)))
        self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.ReLU())
        self.cnn.add(tf.keras.layers.Dropout(.6))
        self.cnn.add(tf.keras.layers.Dense(16 ,kernel_regularizer=tf.keras.regularizers.L2(l2=0.000001)))
        self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.ReLU())
        self.cnn.add(tf.keras.layers.Dropout(.6))
        self.softmax_layer = tf.keras.layers.Dense(4, activation="softmax")

        print(self.cnn.summary())

        if(self.with_loss==True):
            self.vec2veclayer = tf.keras.layers.Dense(2, activation="tanh")



    def call(self, input):

        if(self.without_tempo == True):
            input =input[:,:,:,1:]

        out = self.cnn(input)
        if(self.with_loss==True):

            vec2vec = self.vec2veclayer (out)
            probs = self.softmax_layer(out)
            return probs, vec2vec

        probs = self.softmax_layer(out)
        return probs, probs

    def loss(self, logits, labels,two_d_vectorspace):

        C = our_loss( labels,two_d_vectorspace, self.batch_size)
        my_loss = tf.keras.losses.CategoricalCrossentropy()
        loss =my_loss(labels,logits)
        if(self.with_loss==True):
            loss = loss +C

        return loss



        #return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,logits))

    def accuracy( self,labels, logits ):


        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def train(self, labels,inputs, test_inputs,test_labels):


        num_examples = len(inputs)
        num_batches = int(num_examples/self.batch_size)
        train_acc =[]
        test_acc =[]
        epoch = []

        for epochs in range(self.num_epochs):
            print(epochs)

            indices = np.arange(num_examples)
            indices=tf.random.shuffle(indices)
            shuffled_inputs=  tf.gather(inputs,indices)
            shuffled_labels=  tf.gather(labels,indices)

            for k in range(num_batches):
                step_size=self.batch_size
                batch_inputs =shuffled_inputs[ k*step_size :k*step_size+step_size]
                batch_labels = shuffled_labels[k*step_size :k*step_size+step_size]

                with tf.GradientTape() as tape:
                    logits,two_d_vectorspace= self.call( batch_inputs)
                    loss = self.loss(logits,batch_labels,two_d_vectorspace)
                    print(loss)


                gradients = tape.gradient(loss,self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
            train_acc.append(self.test(labels, inputs))
            test_acc.append(self.test(test_labels, test_inputs))
            epoch.append(epochs)
        return train_acc,test_acc,epoch


    def test(self,labels,inputs):


        num_examples = len(inputs)
        num_batches = int(num_examples/self.batch_size)
        acc=0
        for k in range(num_batches):
            step_size=self.batch_size
            batch_inputs =inputs[ k*step_size :k*step_size+step_size]
            batch_labels = labels[k*step_size :k*step_size+step_size]
            logits,two_d_vectorspace= self.call(batch_inputs)
            acc= acc+self.accuracy(logits,batch_labels)


        return acc/num_batches
