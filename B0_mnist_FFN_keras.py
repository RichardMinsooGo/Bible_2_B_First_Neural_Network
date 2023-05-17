'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

'''
D2. Load MNIST data / Only for Toy Project
'''
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Change data type as float. If it is int type, it might cause error 
'''
D3. Split data
'''
X_train, x_val, Y_train, t_val = \
    train_test_split(X_train, Y_train, test_size=0.2)

'''
D4. Data Preprocessing
'''
# Flattening & Normalizing
X_train = (X_train.reshape(-1, 784) / 255).astype(np.float32)
X_test  = (X_test.reshape(-1, 784) / 255).astype(np.float32)
x_val   = (x_val.reshape(-1, 784) / 255).astype(np.float32)

# One-Hot Encoding
Y_train = np.eye(10)[Y_train].astype(np.float32)
Y_test  = np.eye(10)[Y_test].astype(np.float32)
t_val   = np.eye(10)[t_val].astype(np.float32)

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

hidden_size = 256
output_dim = 10      # output layer dimensionality = num_classes
EPOCHS = 100
batch_size = 100
learning_rate = 5e-4

'''
M3. Build NN model
'''
# model = tf.keras.models.Sequential()
# model.add(Dense(hidden_size, activation='sigmoid'))
# model.add(Dense(hidden_size, activation='sigmoid'))
# model.add(Dense(hidden_size, activation='sigmoid'))
# model.add(Dense(output_dim, activation='softmax'))

class Feed_Forward_Net(Model):
    '''
    Multilayer perceptron
    '''
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.l1 = Dense(hidden_size, activation='sigmoid')
        self.l2 = Dense(hidden_size, activation='sigmoid')
        self.l3 = Dense(hidden_size, activation='sigmoid')
        self.l4 = Dense(output_dim, activation='softmax')

    def call(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        y = self.l4(h3)

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
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

# size = int(len(X_train) * 0.8)
# train_x, val_x = X_train[:size], X_train[size:]
# train_y, val_y = Y_train[:size], Y_train[size:]

'''
M6. Train and Validation - `model.fit`
'''
model.fit(X_train, Y_train,
          epochs=EPOCHS, batch_size=batch_size,
          validation_data=(x_val, t_val))

# model.fit(X_train, Y_train, epochs=30, batch_size=100, verbose=2)
# model.fit(X_train, Y_train, epochs=30, batch_size=100, verbose=2, validation_split=0.2)

'''
M7. Assess model performance
'''
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
    loss,
    acc
))
