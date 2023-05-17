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
D2. Load MNIST data
'''
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

'''
D3. Split data
'''
X_train, x_val, Y_train, t_val = \
    train_test_split(X_train, Y_train, test_size=0.2)

'''
D4. Data Preprocessing
'''
# Flattening
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

hidden_size = 256
output_dim = 10      # output layer dimensionality = num_classes
EPOCHS = 30
batch_size = 100
learning_rate = 5e-4

'''
M3. Build NN model
'''
# model = tf.keras.models.Sequential()
# model.add(Dense(200, activation='sigmoid'))
# model.add(Dense(200, activation='sigmoid'))
# model.add(Dense(200, activation='sigmoid'))
# model.add(Dense(10, activation='softmax'))

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
        out = self.l4(h3)

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

criterion = losses.CategoricalCrossentropy()

train_loss = metrics.Mean()
train_accuracy = metrics.CategoricalAccuracy()

test_loss = metrics.Mean()
test_accuracy = metrics.CategoricalAccuracy()

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
    
    train_loss(loss)
    train_accuracy(labels, predictions)

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
    x_, t_ = shuffle(X_train, Y_train)

    for batch in range(n_batches):
        start = batch * batch_size
        end   = start + batch_size
        train_step(x_[start:end], t_[start:end])

    print('epoch: {}, loss: {:.3}, acc: {:.3f}'.format(
        epoch+1,
        train_loss.result(),
        train_accuracy.result()
    ))

'''
M9. Model evaluation
'''
test_step(X_test, Y_test)

print('test_loss: {:.3f}, test_accuracy: {:.3f}'.format(
    test_loss.result(),
    test_accuracy.result()
))
