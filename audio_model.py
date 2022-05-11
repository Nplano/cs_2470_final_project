import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape



class audio_model(tf.keras.Model):
    def __init__(self):
        super(audio_model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.batch_size = 128
        self.num_epochs = 1
        self.num_classes = 4



        self.cnn = Sequential()
        self.cnn.add(tf.keras.layers.Conv2D(20, kernel_size =(6, 6), strides =(2, 2),activation ='relu',kernel_regularizer=tf.keras.regularizers.L2(l2=0.0000001)))
        self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='valid'))
        self.cnn.add(tf.keras.layers.Conv2D(40, kernel_size =(5, 5), strides =(2, 2),activation ='relu',kernel_regularizer=tf.keras.regularizers.L2(l2=0.0000001)))
        self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='valid'))
        # self.cnn.add(tf.keras.layers.Conv2D(20, kernel_size =(4, 4), strides =(2, 2),activation ='relu'))
        # self.cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='valid'))
        self.cnn.add(Flatten())
        self.cnn.add( tf.keras.layers.Dense(16 ,activation='relu'))
        self.cnn.add(tf.keras.layers.Dropout(.6))

        self.cnn.add( tf.keras.layers.Dense(4 ,activation='softmax'))


    def call(self, input):


        out = self.cnn(input)



        return out ,out

    def loss(self, logits, labels,two_d_vectorspace  ):

        #
        # loss_2d_space =0
        # for i in range(self.batch_size):
        #     label= labels[i]
        #     vector =two_d_vectorspace[i]
        #     if(label[0]==1):
        #         #relaxed bottom right
        #         if(vector[0]<0 and vector[1]<0  ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[0])
        #
        #         if(vector[0]<0 and vector[1]>0 ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[0]+vector[1])
        #
        #         if(vector[0]> 0and vector[1]>0  ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[1])
        #     if(label[1]==1):
        #         #happy top right
        #         if(vector[0]<0 and vector[1]<0  ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[0]+vector[1])
        #
        #         if(vector[0]<0 and vector[1]>0 ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[0])
        #
        #         if(vector[0]> 0and vector[1]<0  ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[1])
        #
        #     if(label[2]==1):
        #         #angry top left
        #         if(vector[0]<0 and vector[1]<0  ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[1])
        #
        #         if(vector[0]>0 and vector[1]<0 ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[0]+vector[1])
        #
        #         if(vector[0]>0 and vector[1]>0  ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[0])
        #
        #
        #     if(label[3]==1):
        #         #sad bottom left
        #         if(vector[0]>0 and vector[1]<0  ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[0])
        #
        #         if(vector[0]<0 and vector[1]>0 ):
        #             loss_2d_space=loss_2d_space+tf.math.square(vector[1])
        #
        #         if(vector[0]>0 and vector[1]>0  ):
        #             loss_2d_space=loss_2d_space+tf.math.square( vector[0]+vector[1])
        #
        #
        #
        #
        # loss_2d_space/=self.batch_size


        #return tf.keras.losses.CategoricalCrossentropy(labels)
        my_loss = tf.keras.losses.CategoricalCrossentropy()
        loss_cross_entropy =my_loss(labels,logits)

        return loss_cross_entropy



        #return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,logits))

    def accuracy( self,labels, logits ):


        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def train(self, labels,inputs):


        num_examples = len(inputs)
        num_batches = int(num_examples/self.batch_size)

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
