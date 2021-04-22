#%% Generating the Dataset
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# %% Reading the Dataset
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# %% Defining the Model
# `nn` is an abbreviation for neural networks
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
# %% Initializing Model Parameters

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# %% Defining the Loss Function
loss = nn.MSELoss()
# %% Defining Optimization Algorithm
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# %% Training
num_epochs = 3
for i in range(num_epochs):
    for X, y in data_iter:
        trainer.zero_grad()
        l = loss(net(X),y)
        l.backward()
        trainer.step()

    l = loss(net(features), labels)
    print(f"Epoch: {i} - Loss: {l:f}")

w = net[0].weight.data
print('error in estimating w:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
# %% How to Access gradients of a weight
print(net[0].weight.grad)

# %%
