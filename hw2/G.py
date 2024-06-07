import needle as ndl
import numpy as np
from needle import nn
from matplotlib import pyplot as plt

A = np.array([[1, 2], [-0.2, 0.5]])
mu = np.array([2, 1])
# total number of sample data to generated
num_sample = 3200
data = np.random.normal(0, 1, (num_sample, 2)) @ A + mu

model_G = nn.Sequential(nn.Linear(2, 2))


def sample_G(model_G, num_samples):
    Z = ndl.Tensor(np.random.normal(0, 1, (num_samples, 2)).astype("float32"))
    fake_X = model_G(Z)
    return fake_X.numpy()


fake_data_init = sample_G(model_G, 3200)

# plt.scatter(data[:, 0], data[:, 1], color="blue", label="real data")
# plt.scatter(fake_data_init[:, 0], fake_data_init[:, 1], color="red", label="G(z) at init")
# plt.legend()
#
# plt.savefig("./pic/image.png")

model_D = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
loss_D = nn.SoftmaxLoss()

opt_G = ndl.optim.Adam(model_G.parameters(), lr=0.01)
def update_G(Z, model_G, model_D, loss_D, opt_G):
    fake_X = model_G(Z)
    fake_Y = model_D(fake_X)
    batch_size = Z.shape[0]
    ones = ndl.ones(batch_size, dtype="int32")
    loss = loss_D(fake_Y, ones)
    loss.backward()
    opt_G.step()

opt_D = ndl.optim.Adam(model_D.parameters(), lr=0.01)

def update_D(X, Z, model_G, model_D, loss_D, opt_D):
    fake_X = model_G(Z).detach()
    fake_Y = model_D(fake_X)
    real_Y = model_D(X)
    assert X.shape[0] == Z.shape[0]
    batch_size = X.shape[0]
    ones = ndl.ones(batch_size, dtype="int32")
    zeros = ndl.zeros(batch_size, dtype="int32")
    loss = loss_D(real_Y, ones) + loss_D(fake_Y, zeros)
    loss.backward()
    opt_D.step()

def train_gan(data, batch_size, num_epochs):
    assert data.shape[0] % batch_size == 0
    for epoch in range(num_epochs):
        begin = (batch_size * epoch) % data.shape[0]
        X = data[begin: begin+batch_size, :]
        Z = np.random.normal(0, 1, (batch_size, 2))
        X = ndl.Tensor(X, dtype="float32")
        Z = ndl.Tensor(Z, dtype="float32")
        update_D(X, Z, model_G, model_D, loss_D, opt_D)
        update_G(Z, model_G, model_D, loss_D, opt_G)

train_gan(data, 32, 2000)

fake_data_trained = sample_G(model_G, 3200)

plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
plt.scatter(fake_data_init[:,0], fake_data_init[:,1], color="red", label="G(z) at init")
plt.scatter(fake_data_trained[:,0], fake_data_trained[:,1], color="pink", label="G(z) trained")

plt.legend()
plt.savefig("./pic/image.png")

print("OK!")
