import process_data as p

import numpy as np

neg_data_path = "./data/MR/rt-polarity.neg"
pos_data_path = "./data/MR/rt-polarity.pos"
save_train_path = "./data/MR/processed/train.txt"
save_test_path = "./data/MR/processed/test.txt"


if __name__ == "__main__":
    # prepare vocabulary
    neg_max_len, neg_words, neg_data = p.get_text(neg_data_path, pos_flag=False)
    pos_max_len, pos_words, pos_data = p.get_text(pos_data_path, pos_flag=True)
    vocab_dict = p.get_vocabulary(neg_words, pos_words)
    max_len = max(neg_max_len, pos_max_len)
    # use word2vector with glove data
    p.get_vector_dict(vocab_dict)
    # get features and labels
    data = np.concatenate((neg_data, pos_data), axis=0)
    # split train and test data
    train_data, test_data = p.split_data(data)
    print(train_data.shape)
    # save processed data
    p.save_processed_data(train_data, save_train_path)
    p.save_processed_data(test_data, save_test_path)