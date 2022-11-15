import  numpy as np
import  torch
import  torch.nn as nn
import  torch.optim as optim
from    matplotlib import pyplot as plt
import math

num_time_steps = 100
input_size = 1
hidden_size = 20
output_size = 1
lr=0.01

class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):

       out, hidden_prev = self.rnn(x, hidden_prev)
       # [b, seq, h]
       out = out.view(-1, hidden_size)
       out = self.linear(out)
       out = out.unsqueeze(dim=0)
       return out, hidden_prev

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

hidden_prev = torch.zeros(1, 1, hidden_size)

for iter in range(3000):
    time_steps = np.linspace(1, 100, num_time_steps)
    np.random.seed(1)
    x = np.random.rand(1)
    y = np.zeros((3, 100))
    y[0, 0] = 1
    y[1, 0] = 0.1 * x
    y[2, 0] = 0
    for i in range(99):
        y[0, i + 1] = y[0, i] + 0.1 * (y[0, i] * y[2, i])
        y[1, i + 1] = y[1, i] + 0.1 * (-y[1, i] * y[2, i])
        y[2, i + 1] = y[2, i] + 0.1 * (-math.pow(y[0, i], 2) + math.pow(y[1, i], 2))
    y = y[1,:]
    y=y.T

    x = torch.tensor(y[:-1]).float().view(1, 99, 1)
    y = torch.tensor(y[1:]).float().view(1, 99, 1)

    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    # for p in model.parameters():
    #     print(p.grad.norm())
    # torch.nn.utils.clip_grad_norm_(p, 10)
    optimizer.step()

    if iter % 100 == 0:
        print("Iteration: {} loss {}".format(iter, loss.item()))

# x = -1 + 2 * np.random.random()
x = 0.5
y = np.zeros((3, 100))
y[0, 0] = 1
y[1, 0] = 0.1 * x
y[2, 0] = 0
for i in range(99):
    y[0, i + 1] = y[0, i] + 0.1 * (y[0, i] * y[2, i])
    y[1, i + 1] = y[1, i] + 0.1 * (-y[1, i] * y[2, i])
    y[2, i + 1] = y[2, i] + 0.1 * (-math.pow(y[0, i], 2) + math.pow(y[1, i], 2))
y=y[1,:]
y = y.T

x = torch.tensor(y[:-1]).float().view(1, 99, 1)
y = torch.tensor(y[1:]).float().view(1, 99, 1)

time_steps = np.linspace(1, 100, num_time_steps)
predictions = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
  input = input.view(1, 1, 1)
  (pred, hidden_prev) = model(input, hidden_prev)
  input = pred
  predictions.append(pred.detach().numpy().ravel()[0])

x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], predictions)
plt.show()

