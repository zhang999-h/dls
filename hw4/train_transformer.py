import time
import sys
import needle as ndl
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cpu()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=256, device=device, dtype="float32")
model = LanguageModel(20, len(corpus.dictionary), hidden_size=32, num_layers=1, seq_model='transformer', seq_len=20, device=device)

start_time = time.time()
train_ptb(model, train_data, seq_len=20, n_epochs=10, device=device, lr=0.003, optimizer=ndl.optim.Adam)
print("Train finished!\nStarting test:")
test_data = ndl.data.batchify(corpus.test, batch_size=16, device=device, dtype="float32")
evaluate_ptb(model, test_data, seq_len=20, device=device)
evaluate_ptb(model, test_data, seq_len=40, device=device)

# 结束计时
end_time = time.time()

# 计算时间差
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")