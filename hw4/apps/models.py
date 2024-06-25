import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(self, c_in, c_out, k, s, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.layer = nn.Sequential(
            nn.Conv(c_in, c_out, k, s, device=device, dtype=dtype),
            nn.BatchNorm2d(c_out, device=device, dtype=dtype),
            nn.ReLU()
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.layer(x)
        ### END YOUR SOLUTION


class ResidualBlock(ndl.nn.Module):
    def __init__(self, c_in, c_out, k, s, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        m = nn.Sequential(
            ConvBN(c_in, c_out, k, s, device=device, dtype=dtype),
            ConvBN(c_in, c_out, k, s, device=device, dtype=dtype)
        )
        self.layer = nn.Residual(m)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.layer(x)
        ### END YOUR SOLUTION


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.conv_bn1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.conv_bn2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)
        self.res1 = ResidualBlock(32, 32, 3, 1, device=device, dtype=dtype)
        self.conv_bn3 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.conv_bn4 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)
        self.res2 = ResidualBlock(128, 128, 3, 1, device=device, dtype=dtype)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        out = self.conv_bn1(x)
        out = self.conv_bn2(out)
        out = self.res1(out)
        out = self.conv_bn3(out)
        out = self.conv_bn4(out)
        out = self.res2(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.seq_model = seq_model
        self.embedding_layer = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model is 'rnn':
            self.seq_layer = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        else:
            self.seq_layer = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.linear_layer = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size = x.shape
        x_embedding = self.embedding_layer(x)
        out, h = self.seq_layer(x_embedding, h)
        out = self.linear_layer(out.reshape(seq_len * batch_size, self.hidden_size))
        # 返回的是 out(seq_len*bs, output_size) output_size是one_hot长度，
        # 所以out相当于所有预测的单词的one_hot
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)