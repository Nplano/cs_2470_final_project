import numpy as np

def get_labels(labels_text_file):
    """
    :param labels_text_file: path to mood labels (data/moods.txt)
    :return one_hot: one hot encoded labels according to scheme:
        relaxed = [1 0 0 0]
        happy = [0 1 0 0]
        angry = [0 0 1 0]
        sad = [0 0 0 1]
    """
    with open(labels_text_file) as f:
        text_file = f.readlines()

    one_hot = []
    i = 0
    for item in text_file:
        if i not in {1087, 2043, 2174, 2212, 2240, 2301, 2359, 2374, 2491}:
            string = item.strip('\n')
            if(string == 'relaxed'):
                one_hot.append([1, 0, 0, 0])
            elif(string == 'happy'):
                one_hot.append([0, 1, 0, 0])
            elif(string == 'angry'):
                one_hot.append([0, 0, 1, 0])
            elif(string == 'sad'):
                one_hot.append([0,0, 0, 1])
        i += 1
    one_hot = np.array(one_hot)
    return one_hot
