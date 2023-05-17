'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split

'''
D2. Generate make_moon data
'''
N = 4000
x, t = datasets.make_moons(N, noise=0.3)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,4))
plt.scatter(x[:,0], x[:,1], c=t, cmap=plt.cm.winter)

t = t.reshape(N, 1)

'''
D3. Split data
'''
X_train, X_test, Y_train, Y_test = \
    train_test_split(x, t, test_size=0.2)


'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optimizers


np.random.seed(123)
torch.manual_seed(123)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
M2. Set Hyperparameters
'''

input_size = 2  # input layer dimensionality
hidden_size = 10
output_dim = 1  # output layer dimensionality
EPOCHS = 100
batch_size = 100
learning_rate = 0.1

'''
M3. DataLoader
'''
# No need for make_moons, it is very small size data
'''
M4. Build NN model
'''
class Feed_Forward_Net(nn.Module):
    '''
    Multilayer perceptron
    '''
    def __init__(self, input_size, hidden_size, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.a1 = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, output_dim)
        self.a2 = nn.Sigmoid()

        self.layers = [self.l1, self.a1, self.l2, self.a2]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

model = Feed_Forward_Net(input_size, hidden_size, output_dim).to(device)

'''
M5. Optimizer
'''

optimizer = optimizers.SGD(model.parameters(), lr=learning_rate)

'''
M6. Define Loss Fumction
'''

criterion = nn.BCELoss()
def compute_loss(t, y):
    return criterion(y, t)

'''
M7. Define train loop
'''

def train_step(images, labels):
    model.train()

    # Forward pass
    predictions = model(images)
    loss = compute_loss(labels, predictions)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

'''
M8. Define validation / test loop
'''

def test_step(images, labels):
    images = torch.Tensor(images).to(device)
    labels = torch.Tensor(labels).to(device)
    model.eval()
    predictions = model(images)
    loss = compute_loss(labels, predictions)

    return loss, predictions

'''
M9. Define Episode / each step process
'''

n_batches = X_train.shape[0] // batch_size

for epoch in range(EPOCHS):
    train_loss = 0.
    x_, t_ = shuffle(X_train, Y_train)
    x_ = torch.Tensor(x_).to(device)
    t_ = torch.Tensor(t_).to(device)

    for batch in range(n_batches):
        start = batch * batch_size
        end   = start + batch_size
        loss  = train_step(x_[start:end], t_[start:end])
        train_loss += loss.item()/n_batches

    print('epoch: {}, loss: {:.3f}'.format(
        epoch+1,
        train_loss
    ))

'''
M10. Model evaluation
'''
loss, preds = test_step(X_test, Y_test)
test_loss = loss.item()
preds = preds.data.cpu().numpy() > 0.5
test_acc = accuracy_score(Y_test, preds)

print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
    test_loss,
    test_acc
))
