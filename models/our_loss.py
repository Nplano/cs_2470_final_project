import numpy as np
import tensorflow as tf





def our_loss( labels,two_d_vectorspace, batch_size):

    loss_2d_space =0
    for i in range(batch_size):
        label= labels[i]
        vector =two_d_vectorspace[i]
        if(label[0]==1):
            #relaxed bottom right
            if(vector[0]<0 and vector[1]<0  ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[0])

            if(vector[0]<0 and vector[1]>0 ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[0]+vector[1])

            if(vector[0]> 0and vector[1]>0  ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[1])
            #loss_2d_space=loss_2d_space+distance(vector[1],vector[1])
        if(label[1]==1):
            #happy top right
            if(vector[0]<0 and vector[1]<0  ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[0]+vector[1])

            if(vector[0]<0 and vector[1]>0 ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[0])

            if(vector[0]> 0and vector[1]<0  ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[1])

        if(label[2]==1):
            #angry top left
            if(vector[0]<0 and vector[1]<0  ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[1])

            if(vector[0]>0 and vector[1]<0 ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[0]+vector[1])

            if(vector[0]>0 and vector[1]>0  ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[0])

        if(label[3]==1):
            #sad bottom left
            if(vector[0]>0 and vector[1]<0  ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[0])

            if(vector[0]<0 and vector[1]>0 ):
                loss_2d_space=loss_2d_space+tf.math.square(1+vector[1])

            if(vector[0]>0 and vector[1]>0  ):
                loss_2d_space=loss_2d_space+tf.math.square( 1+vector[0]+vector[1])

    loss_2d_space/=batch_size

    return loss_2d_space

def distance(x1,y1,x2,y2):
    return tf.math.square(x2-x1)+tf.math.square(y2-y1)
