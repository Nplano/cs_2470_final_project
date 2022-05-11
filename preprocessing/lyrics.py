import numpy as np
from collections import defaultdict


PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
MAX_SENTENCE_SIZE = 150  # the max length of each sentence/song - feel free to adjust
UNK_THRESHOLD = 2  # words showing up less than this threshold will be considered as uncommon words

def read_data(file_name):
    """
    Load text data from file
    :param file_name:  string, name of data file
    :return: list of sentences, each a list of words split on whitespace
    """
    text = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if (i != 1087 and i != 2174 and i != 2212 and
                i != 2240 and i != 2301 and i != 2359 and i != 2374 and i != 2491 and i != 2043):
            text.append(lines[i].split())

    return text


def mark_UNK(sentences):
    """
    mark uncommon words as *UNK*
    :param sentences:  list of lists of words, each representing padded sentence
    :return: sentences with marked words
    """
    freq_dict = defaultdict(int)
    for sentence in sentences:
        words = set(sentence)
        for word in words:
            freq_dict[word] += 1

    uncommon_words = set()
    for word, frequency in freq_dict.items():
        if frequency < UNK_THRESHOLD:
            uncommon_words.add(word)

    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            if sentences[i][j] in uncommon_words:
                sentences[i][j] = UNK_TOKEN


def build_vocab(sentences):
    """
    Builds vocab from list of sentences
    :param sentences:  list of sentences, each a list of words
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
    tokens = []
    for s in sentences: tokens.extend(s)
    all_words = sorted(list(set([STOP_TOKEN, PAD_TOKEN, UNK_TOKEN] + tokens)))

    vocab = {word: i for i, word in enumerate(all_words)}

    return vocab, vocab[PAD_TOKEN]


def pad_corpus(sentences):
    """
    Returns padded sentences of a fixed length
    :param sentences:  list of lists of words, each representing padded sentence
    :return: list of padded sentences
    """
    padded_sentences = []
    for line in sentences:
        padded_line = line[:MAX_SENTENCE_SIZE]
        padded_line += [STOP_TOKEN] + [PAD_TOKEN] * (MAX_SENTENCE_SIZE - len(padded_line))
        padded_sentences.append(padded_line)

    return padded_sentences


def process_missing_lyrics(sentences):
    """
    Delete blank lines in lyrics and Returns the indices of missing lyrics
    :param sentences:  list of lists of words, each representing padded sentence
    :return: list of indices of missing lyrics
    """
    missing_lyric_indices = []
    for i in range(len(sentences)):
        line = sentences[i]
        if len(line) == 0:
            missing_lyric_indices.append(i)

    for i in range(len(missing_lyric_indices)):
        sentences.remove([])

    return [int(item) for item in missing_lyric_indices]


def convert_to_id(vocab, sentences):
    """
    Convert sentences to indexed
    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    ids = [[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences]
    return np.stack(ids)


def get_lyric_data(lyrics_text_file):
    sentences = read_data(lyrics_text_file)
    lyric_missing_indices = process_missing_lyrics(sentences)

    mark_UNK(sentences)

    sentences = pad_corpus(sentences)

    vocab, pad_token_ind = build_vocab(sentences)
    lyrics_ids = convert_to_id(vocab, sentences)

    # 80% go to training set and 20% go to testing set
    training_size = int(lyrics_ids.shape[0] * 0.80)
    training_lyrics_ids = lyrics_ids[:training_size]
    testing_lyrics_ids = lyrics_ids[training_size:]

    return training_lyrics_ids, testing_lyrics_ids, vocab, lyric_missing_indices, pad_token_ind


def lyric_labels_process(one_hot_labels, lyric_missing_indices):
    one_hot_labels = np.delete(one_hot_labels, lyric_missing_indices, axis=0)

    training_size = int(one_hot_labels.shape[0] * 0.80)
    training_labels = one_hot_labels[:training_size]
    testing_labels = one_hot_labels[training_size:]
    return training_labels, testing_labels
