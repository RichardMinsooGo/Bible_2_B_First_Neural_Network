'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import numpy as np

import os
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

'''
D2. Load MNIST data
'''
root = os.path.join('~', '.torch', 'mnist')
transform = transforms.Compose([transforms.ToTensor(),
                                lambda x: x.view(-1)])
train_dataset = datasets.MNIST(root=root,
                             download=True,
                             train=True,
                             transform=transform)
test_dataset = datasets.MNIST(root=root,
                            download=True,
                            train=False,
                            transform=transform)

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optimizers


np.random.seed(123)
torch.manual_seed(123)

'''
M2. Set Hyperparameters
'''
input_size = 784 # 28x28
hidden_size = 256 
output_dim = 10 # output layer dimensionality = num_classes
EPOCHS = 30
batch_size = 100
learning_rate = 5e-4

'''
M3. DataLoader
'''

train_ds = torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size, 
                                       shuffle=True)
test_ds = torch.utils.data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size, 
                                      shuffle=False)

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
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.a2 = nn.Sigmoid()
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.a3 = nn.Sigmoid()
        self.l4 = nn.Linear(hidden_size, output_dim)

        self.layers = [self.l1, self.a1,
                       self.l2, self.a2,
                       self.l3, self.a3,
                       self.l4]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

'''
M5. Transfer model to GPU
'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Feed_Forward_Net(input_size, hidden_size, output_dim).to(device)

'''
M6. Optimizer
'''
# optimizer = optimizers.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

'''
M7. Define Loss Fumction
'''

criterion = nn.CrossEntropyLoss()
def compute_loss(t, y):
    return criterion(y, t)

'''
M8. Define train loop
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

    return loss, predictions

'''
M9. Define validation / test loop
'''

def test_step(images, labels):
    model.eval()
    predictions = model(images)
    loss = compute_loss(labels, predictions)

    return loss, predictions

'''
M10. Define Episode / each step process
'''

for epoch in range(EPOCHS):
    train_loss = 0.
    train_acc = 0.

    for (x, t) in train_ds:
        x, t = x.to(device), t.to(device)
        loss, preds = train_step(x, t)
        train_loss += loss.item()
        train_acc += \
            accuracy_score(t.tolist(),
                           preds.argmax(dim=-1).tolist())

    train_loss /= len(train_ds)
    train_acc /= len(train_ds)

    print('epoch: {}, loss: {:.3}, acc: {:.3f}'.format(
        epoch+1,
        train_loss,
        train_acc
    ))

'''
M11. Model evaluation
'''

test_loss = 0.
test_acc = 0.

for (x, t) in test_ds:
    x, t = x.to(device), t.to(device)
    loss, preds = test_step(x, t)
    test_loss += loss.item()
    test_acc += \
        accuracy_score(t.tolist(),
                       preds.argmax(dim=-1).tolist())

test_loss /= len(test_ds)
test_acc /= len(test_ds)
print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
    test_loss,
    test_acc
))
