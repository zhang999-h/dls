import sys
import time
sys.path.append('./python')
import needle as ndl
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cuda()
corpus = ndl.data.Corpus("data/ptb")
start_time = time.time()
# 58099,16
train_data = ndl.data.batchify(corpus.train, batch_size=16, device=device, dtype="float32")
model = LanguageModel(30, len(corpus.dictionary), hidden_size=10, num_layers=2, seq_model='lstm', device=device)
train_ptb(model, train_data, seq_len=100, n_epochs=1, device=device)
print("Train finished!\nStarting test:")
test_data = ndl.data.batchify(corpus.test, batch_size=16, device=device, dtype="float32")
evaluate_ptb(model, test_data, seq_len=40, device=device)

# 结束计时
end_time = time.time()

# 计算时间差
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
