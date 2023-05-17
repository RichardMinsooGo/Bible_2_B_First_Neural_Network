'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import numpy as np
import tensorflow as tf
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

np.random.seed(123)
tf.random.set_seed(123)

'''
M2. Set Hyperparameters
'''

hidden_size = 10
output_dim = 1  # output layer dimensionality
EPOCHS = 100
batch_size = 100
learning_rate = 0.1

'''
M3. Build NN model
'''
class Feed_Forward_Net(Model):
    '''
    Multilayer perceptron
    '''
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.l1 = Dense(hidden_size, activation='sigmoid')
        self.l2 = Dense(output_dim, activation='sigmoid')

    def call(self, x):
        h = self.l1(x)
        out = self.l2(h)

        return out

model = Feed_Forward_Net(hidden_size, output_dim)

'''
M4. Optimizer
'''

optimizer = optimizers.SGD(learning_rate=learning_rate)
# optimizer = optimizers.Adam(learning_rate=learning_rate)

'''
M5. Define Loss Fumction
'''

criterion = losses.BinaryCrossentropy()
test_loss = metrics.Mean()
test_accuracy  = metrics.BinaryAccuracy()

def compute_loss(t, y):
    return criterion(t, y)

'''
M6. Define train loop
'''

def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = compute_loss(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

'''
M7. Define validation / test loop
'''

def test_step(images, labels):
    predictions = model(images)
    t_loss = compute_loss(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

    return t_loss


'''
M8. Define Episode / each step process
'''

n_batches = X_train.shape[0] // batch_size

for epoch in range(EPOCHS):
    train_loss = 0.
    x_, t_ = shuffle(X_train, Y_train)

    for batch in range(n_batches):
        start = batch * batch_size
        end   = start + batch_size
        loss  = train_step(x_[start:end], t_[start:end])
        train_loss += loss.numpy()/n_batches

    print('epoch: {}, loss: {:.3}'.format(
        epoch+1,
        train_loss
    ))

'''
M9. Model evaluation
'''
test_step(X_test, Y_test)

print('test_loss: {:.3f}, test_accuracy: {:.3f}'.format(
    test_loss.result(),
    test_accuracy.result()
))
