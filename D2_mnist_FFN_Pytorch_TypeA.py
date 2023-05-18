'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import numpy as np

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

'''
D2. Load MNIST data
'''
train_dataset = datasets.MNIST(root='./data', 
                             download=True,
                             train=True,
                             transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root='./data', 
                            download=True,
                            train=False,
                             transform=transforms.ToTensor())

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

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
learning_rate = 0.001

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
# Other FFN
class Feed_Forward_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim):
        super(Feed_Forward_Net, self).__init__()
        self.input_size = input_size
        self.layer_1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer_2 = nn.Linear(hidden_size, output_dim)  
    
    def forward(self, x):
        x   = self.layer_1(x)
        x   = self.relu(x)
        x   = self.dropout1(x)
        out = self.layer_2(x)
        # no activation and no softmax at the end
        return out
'''

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

'''
M8. Define train loop
'''

def train_step(model, images, labels):
    model.train()
    # origin shape: [100, 1, 28, 28]
    # resized: [100, 784]
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    # Forward pass
    predictions = model(images)
    loss = criterion(predictions, labels)
    loss_val = loss.item()

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Pytorch need a manual coding for accuracy
    # max returns (value ,index)
    _, predicted = torch.max(predictions.data, 1)           
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    
    return loss_val, acc

'''
M9. Define validation / test loop
'''

def test_step(model, images, labels):
    model.eval()
    # origin shape: [100, 1, 28, 28]
    # resized: [100, 784]
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    # Forward pass
    predictions = model(images)
    loss = criterion(predictions, labels)
    loss_val = loss.item()

    # Pytorch need a manual coding for accuracy
    # max returns (value ,index)
    _, predicted = torch.max(predictions.data, 1)           
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    
    return loss_val, acc

'''
M10. Define Episode / each step process
'''
from tqdm import tqdm, tqdm_notebook, trange

for epoch in range(EPOCHS):
    
    with tqdm_notebook(total=len(train_ds), desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        
        for i, (images, labels) in enumerate(train_ds):
         
            loss_val, acc = train_step(model, images, labels)
            
            train_losses.append(loss_val)
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")


'''
M11. Model evaluation
'''
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():

    with tqdm_notebook(total=len(test_ds), desc=f"Test_ Epoch {epoch+1}") as pbar:    
        test_losses = []
        test_accuracies = []

        for images, labels in test_ds:
            loss_val, acc = test_step(model, images, labels)

            test_losses.append(loss_val)
            test_accuracies.append(acc)

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")
            
