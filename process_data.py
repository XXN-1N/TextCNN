import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter


MAX_VOCAB_SIZE = 16000
test_ratio = 0.1

glove_path = "./data/glove/glove.6B.100d.txt"
neg_data_path = "./data/MR/rt-polarity.neg"
pos_data_path = "./data/MR/rt-polarity.pos"
save_train_path = "./data/MR/processed/train.txt"
save_test_path = "data/MR/processed/test.txt"
save_glove_path = "./data/glove/glove_data_6B_100d.pkl"


def get_vector_dict(vocab_dict):
    words = ['<PAD>', '<UNK>']
    embeds = np.zeros(shape=[2, 100], dtype=np.float32)
    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            if values[0] in vocab_dict.keys():
                words.append(values[0])
                vector = np.asarray(values[1:], "float32")
                embeds = np.concatenate((embeds, vector.reshape(1, 100)), axis=0)

    w2idx_dict = dict(zip(words, range(len(words))))
    idx2w_dict = dict(zip(range(len(words)), words))
    assert embeds.shape[0] == len(words)

    glove_data = {'w2idx_dict': w2idx_dict,
                  'idx2w_dict': idx2w_dict,
                  'embed_matrix': embeds,
                  }
    save_pkl_file(glove_data, save_glove_path)


def get_text(file_path, pos_flag):
    words, text, label = (), [], []
    max_len = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.lower().split()
            text.append(np.array(line))
            words += tuple(line)
            max_len = max(max_len, len(line))

    if pos_flag:
        label = np.zeros((len(text), 1), dtype=np.int)
    else:
        label = np.ones((len(text), 1), dtype=np.int)

    text = np.array(text)[:, np.newaxis]
    labeled_data = np.concatenate((text, label), axis=1)
    return max_len, words, labeled_data


def get_vocabulary(neg_words, pos_words):
    words = neg_words + pos_words
    vocab_dict = dict(Counter(words).most_common(MAX_VOCAB_SIZE - 1))
    return vocab_dict


def split_data(data):
    np.random.seed(43)
    shuffled_indices = np.random.permutation(data.shape[0])
    test_set_size = int(data.shape[0] * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices, :], data[test_indices, :]


def save_pkl_file(data, path):
    pkl_output = open(path, "wb")
    pickle.dump(data, pkl_output)
    pkl_output.close()


def save_processed_data(data, filepath):
    with open(filepath, "w+", encoding="utf-8") as f:
        for line in data:
            f.write(str(line))
            f.write("\n")


def padding(data, max_len):
    X_data, y = data[:, 0], data[:, 1]
    # seq_len = [len(i) for i in batch_data]
    padding_data = []
    for seq in X_data:
        pad_len = max_len - len(seq)
        seq = seq[np.newaxis, :]
        if pad_len != 0:
            pad_seq = np.array([['<PAD>'] * pad_len])
            seq = np.concatenate((seq, pad_seq), axis=1)
        padding_data.append(seq)
    padding_data = np.squeeze(padding_data)
    y = y[:, np.newaxis]
    padding_data = np.concatenate((padding_data, y), axis=1)
    return padding_data


def plot_fig(train_loss, train_acc, test_loss, test_acc):
    fig = plt.figure(figsize=(20, 20))  # 建立一个大小为10*8的画板
    ax1 = fig.add_subplot(211)  # 在画板上添加3*3个画布，位置是第1个
    ax1.plot(range(len(train_acc)), train_acc, 'b', label='Training accuracy')
    ax1.plot(range(len(train_loss)), train_loss, 'r', label='Training loss')
    ax1.set_title('Training accuracy & loss')

    ax2 = fig.add_subplot(212)
    ax2.plot(range(len(test_acc)), test_acc, 'b', label='Validation accuracy')
    ax2.plot(range(len(test_loss)), test_loss, 'r', label='Validation loss')
    ax2.set_title('Validation accuracy & loss')

    plt.figure()
    plt.show()

