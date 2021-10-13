import process_data as p
import model

import torch
import numpy as np
import pickle
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


batch_size = 64
lr = 1e-3
EPOCH = 50
neg_data_path = "./data/MR/rt-polarity.neg"
pos_data_path = "./data/MR/rt-polarity.pos"
save_train_path = "./data/MR/processed/train.txt"
save_test_path = "./data/MR/processed/test.txt"
save_glove_path = "./data/glove/glove_data_6B_100d.pkl"


class CNNDataset(Dataset):
    def __init__(self, data):
        self.X_data = data[:, : -1]
        self.y_data = data[:, -1]
        self.embed_vectors, self.w2idx_dict = self._load_vector()

    def __len__(self):
        assert len(self.X_data) == self.y_data.shape[0]
        return len(self.X_data)

    def __getitem__(self, idx):
        X = self.X_data[idx]
        w_idx = []
        for i in X:
            if i in self.w2idx_dict.keys():
                w_idx.append(self.w2idx_dict[i])
            else:
                w_idx.append(self.w2idx_dict['<UNK>'])
        vector_line = self.embed_vectors[w_idx]
        return vector_line, self.y_data[idx]

    @staticmethod
    def _load_vector():
        pkl_input = open(save_glove_path, "rb")
        glove_data = pickle.load(pkl_input)
        embed_vectors = glove_data['embed_matrix']
        w2idx_dict = glove_data['w2idx_dict']
        return np.array(embed_vectors), w2idx_dict


def load_data(path):
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(line.strip())
    return np.array(data_list)


def train(my_model, loader):
    my_model.train()
    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(my_model.parameters(), lr=lr)
    '''
    1. 样本总数/batch_size是走完一个epoch所需的“步数”=len(loader)
    2. len(loader.dataset)是样本总数
    '''
    epoch_loss, acc = 0, 0
    for step, (X, y) in tqdm(enumerate(loader), desc='Training', total=len(loader)):
        y_pred = my_model(X)
        loss = criterion(y_pred, y)     # 计算了一个mini_batch的均值了，因此累加以后需要除以的步数
        epoch_loss += loss.item()
        y_pred = y_pred.argmax(dim=1)
        acc += torch.eq(y_pred, y).sum().item()
        optimizer.zero_grad()   # 清除上一次计算的梯度
        loss.backward()     # 计算梯度
        optimizer.step()    # 更新权重

    return model, epoch_loss/len(loader), acc/len(loader.dataset)


def eval(epoch, my_model, loader):
    dev_epoch_loss, acc = 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for step, (X, y) in enumerate(loader):
            y_pred = my_model(X)
            loss = criterion(y_pred, y)
            dev_epoch_loss = loss.item()
            y_pred = y_pred.argmax(dim=1)
            acc += torch.eq(y_pred, y).sum().item()

        print('Epoch:{} -- Loss:{} -- Accuracy:{}\n'.format(epoch, dev_epoch_loss/len(loader), acc/len(loader.dataset)))
        return model, dev_epoch_loss/len(loader), acc/len(loader.dataset)


if __name__ == "__main__":
    logging.info('Phase 1: Data Preparation')
    # prepare vocabulary
    neg_max_len, neg_words, neg_data = p.get_text(neg_data_path, pos_flag=False)
    pos_max_len, pos_words, pos_data = p.get_text(pos_data_path, pos_flag=True)
    vocab_dict = p.get_vocabulary(neg_words, pos_words)
    max_len = max(neg_max_len, pos_max_len)
    # use word2vector with glove data
    p.get_vector_dict(vocab_dict)
    # get features and labels
    all_data = np.concatenate((neg_data, pos_data), axis=0)
    # padding
    data = p.padding(all_data, max_len)
    # split train and test data
    train_data, test_data = p.split_data(data)

    # save processed data
    p.save_processed_data(train_data, save_train_path)
    p.save_processed_data(test_data, save_test_path)

    # dataset
    train_dataset = CNNDataset(train_data)
    test_dataset = CNNDataset(test_data)
    # dataloader
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True)
    # load model
    model = model.TextCNN(embedding_dim=100, in_channels=1, out_channels=2,
                          num_kernel=100, kernel_size=[3, 4, 5], dropout=0.5)

    logging.info('Phase 2: Model Training & Validation')
    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []
    for i in range(EPOCH):
        model, train_loss, train_acc = train(model, train_data_loader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        model, test_loss, test_acc = eval(i, model, test_data_loader)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    p.plot_fig(train_loss_list, train_acc_list, test_loss_list, test_acc_list)