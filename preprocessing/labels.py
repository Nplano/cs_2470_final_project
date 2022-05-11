

from matplotlib import pyplot as plt
import numpy as np


def get_labels(labels_text_file):

    # takes in labels file and 1 hot encodes them with the following encoding
    # text file is moods.txt
    #relaxed = [1 0 0 0]
    #happy = [0 1 0 0]
    # angry = [0 0 1 0]
    #sad = [0 0 0 1]



    with open(labels_text_file) as f:
        text_file = f.readlines()

    one_hot =[]
    i=0
    for item in text_file:
        i=i+1

        if(i !=1087 and i !=2174 and i !=2212 and
        i !=2240 and i !=2301 and i !=2359 and i !=2374 and i !=2491 and i !=2043):
            string =item.strip('\n')

            if(string =='angry'):
                one_hot.append([0, 0, 1, 0])
            elif(string =='happy'):
                one_hot.append([0, 1, 0, 0])
            elif(string =='sad'):
                one_hot.append([0,0, 0, 1])
            elif(string =='relaxed'):
                one_hot.append([1, 0, 0, 0])

    one_hot = np.array(one_hot)

    # output tuple numpy array with dimensions [number of songs, 4] and 2d vector space


    return one_hot
