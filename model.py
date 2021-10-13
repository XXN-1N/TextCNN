import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, embedding_dim, in_channels, out_channels, num_kernel, kernel_size, dropout):
        super(TextCNN, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_kernel,
                      kernel_size=(h, embedding_dim))
            for h in kernel_size
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=len(kernel_size) * num_kernel,
                                out_features=out_channels,
                                bias=True)

    def forward(self, X):
        # print(X.shape)   # torch.Size([64, 59, 100])
        X = X.unsqueeze(1)
        # torch.Size([64, 1, 59, 100])
        X = [conv(X) for conv in self.convs]
        # print(X[1].shape)   # torch.Size([64, 100, 56, 1])
        X = [i.squeeze(3) for i in X]
        # print(X[1].shape)   # torch.Size([64, 100, 56])
        # print(X[1].shape)     # torch.Size([64, 100, 56])
        X = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in X]
        # print(X[1].shape)     # torch.Size([64, 100])
        X = torch.cat(X, dim=1)
        X = self.dropout(X)
        X = self.fc(X)
        return X
