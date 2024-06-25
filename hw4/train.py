import sys

import needle as ndl
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cpu()
corpus = ndl.data.Corpus("data/ptb")
# 58099,16
train_data = ndl.data.batchify(corpus.train, batch_size=16, device=ndl.cpu(), dtype="float32")
model = LanguageModel(30, len(corpus.dictionary), hidden_size=10, num_layers=2, seq_model='rnn', device=ndl.cpu())
train_ptb(model, train_data, seq_len=40, n_epochs=1, device=device)
evaluate_ptb(model, train_data, seq_len=40, device=device)