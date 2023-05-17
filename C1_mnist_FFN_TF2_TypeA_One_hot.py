'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import tensorflow as tf

'''
D2. Load MNIST data / Only for Toy Project
'''

# print(tf.__version__)
## MNIST Dataset #########################################################
mnist = tf.keras.datasets.mnist
# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
##########################################################################

## Fashion MNIST Dataset #################################################
# mnist = tf.keras.datasets.fashion_mnist
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
##########################################################################
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Change data type as float. If it is int type, it might cause error 
'''
D3. Data Preprocessing
'''
# Normalizing
X_train, X_test = X_train / 255.0, X_test / 255.0

print(Y_train[0:10])
print(X_train.shape)

# One-Hot Encoding
from tensorflow.keras.utils import to_categorical

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

'''
D4. EDA(? / Exploratory data analysis)
'''
import matplotlib.pyplot as plt

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # if you want to invert color, you can use 'gray_r'. this can be used only for MNIST, Fashion MNIST not cifar10
    # pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray_r'))
    
# show the figure
plt.show()

'''
D5. Build dataset
'''

# in the case of Keras or TF2, type shall be [image_size, image_size, 1]
# if it is RGB type, type shall be [image_size, image_size, 3]
# For MNIST or Fashion MNIST, it need to reshape
X_train_new_axis = X_train[..., tf.newaxis]
X_test_new_axis = X_test[..., tf.newaxis]

print(X_train_new_axis.shape)
    
# 입력된 buffer_size만큼 data를 채우고 무작위로 sampling하여 새로운 data로 바꿉니다.
# 완벽한 셔플링을 위해서는 데이터 세트의 전체 크기보다 크거나 같은 버퍼 크기가 필요합니다.
# 만약 작은 데이터수보다 작은 buffer_size를 사용할경우,
# 처음에 설정된 buffer_size만큼의 data안에서 임의의 셔플링이 발생합니다.
shuffle_size = 100000

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train_new_axis, Y_train)).shuffle(shuffle_size).batch(batch_size)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test_new_axis, Y_test)).batch(batch_size)

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

'''
M2. Set Hyperparameters
'''

hidden_size = 256
output_dim = 10      # output layer dimensionality = num_classes
EPOCHS = 100
batch_size = 100
learning_rate = 0.001
'''
M3. Build NN model
'''
# Fully connected neural network with one hidden layer
class Feed_Forward_Net(Model):
    def __init__(self):
        super(Feed_Forward_Net, self).__init__()
        self.flatten = Flatten()
        self.layer_1 = Dense(hidden_size, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.layer_2 = Dense(output_dim, activation='softmax')

    def call(self, x):
        x   = self.flatten(x)
        x   = self.layer_1(x)
        x   = self.dropout1(x)
        out = self.layer_2(x)
        return out

model = Feed_Forward_Net()
# model.summary()

'''
M4. Optimizer
'''

optimizer = tf.keras.optimizers.Adam()

'''
M5. Define Loss Fumction
'''

criterion = losses.CategoricalCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = metrics.Mean(name='test_loss')
test_accuracy = metrics.CategoricalAccuracy(name='test_accuracy')

'''
M6. Define train loop
'''

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = criterion(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)
    

'''
M7. Define validation / test loop
'''

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = criterion(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

'''
M8. Define Episode / each step process
'''

from tqdm import tqdm, tqdm_notebook, trange

for epoch in range(EPOCHS):
    
    with tqdm_notebook(total=len(train_ds), desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        
        for images, labels in train_ds:
            
            train_step(images, labels)
            
            loss_val= train_loss.result()
            acc     = train_accuracy.result()*100
            
            train_losses.append(loss_val)
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")
            
'''
M9. Model evaluation
'''
with tqdm_notebook(total=len(test_ds), desc=f"Test_ Epoch {epoch+1}") as pbar:    
    test_losses = []
    test_accuracies = []
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

        loss_val= test_loss.result()
        acc     = test_accuracy.result()*100

        test_losses.append(loss_val)
        test_accuracies.append(acc)

        pbar.update(1)
        pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")
            
