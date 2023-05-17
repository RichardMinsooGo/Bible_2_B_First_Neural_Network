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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

np.random.seed(123)
tf.random.set_seed(123)


'''
M2. Set Hyperparameters
'''

hidden_size = 10
output_dim = 1  # output layer dimensionality = num_classes
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
        y = self.l2(h)

        return y

model = Feed_Forward_Net(hidden_size, output_dim)

'''
M4. Optimizer
'''

optimizer = optimizers.SGD(learning_rate=learning_rate)
# optimizer = optimizers.Adam(learning_rate=learning_rate)

'''
M5. Model Compilation - model.compile
'''
model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])

'''
M6. Train and Validation - `model.fit`
'''
model.fit(X_train, Y_train,
          epochs=EPOCHS, batch_size=batch_size,
          verbose=1)

'''
M7. Assess model performance
'''
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
    loss,
    acc
))
