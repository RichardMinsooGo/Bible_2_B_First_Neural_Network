'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import tensorflow as tf

import matplotlib.pyplot as plt
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

'''
D4. EDA(? / Exploratory data analysis)
'''

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
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''
from tensorflow.keras.layers import Dense , Flatten, Dropout
'''
M2. Set Hyperparameters
'''

hidden_size = 256
output_dim = 10      # output layer dimensionality = num_classes
EPOCHS = 30
'''
M3. Build NN model
'''
# in the case of Keras or TF2, type shall be [image_size, image_size, 1]
model = tf.keras.models.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(hidden_size, activation='relu'),
    Dropout(0.2),
    Dense(output_dim, activation='softmax')
])

model.summary()
'''
M4. Optimizer
'''
# Optimizer can be included at model.compile

'''
M5. Model Compilation - model.compile
'''

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''
M6. Train and Validation - `model.fit`
'''

model.fit(X_train, Y_train, epochs=EPOCHS, verbose=1)

'''
M7. Assess model performance
'''
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
    loss,
    acc
))

'''
M8. [Opt] Assess model performance
'''
model.evaluate(X_test,  Y_test, verbose=2)              


'''

Keras Display Options

verbose default is 1

verbose=0 (silent)

verbose=1 (progress bar)

Train on 186219 samples, validate on 20691 samples
Epoch 1/2
186219/186219 [==============================] - 85s 455us/step - loss: 0.5815 - acc: 
0.7728 - val_loss: 0.4917 - val_acc: 0.8029
Train on 186219 samples, validate on 20691 samples
Epoch 2/2
186219/186219 [==============================] - 84s 451us/step - loss: 0.4921 - acc: 
0.8071 - val_loss: 0.4617 - val_acc: 0.8168


verbose=2 (one line per epoch)

Train on 186219 samples, validate on 20691 samples
Epoch 1/1
 - 88s - loss: 0.5746 - acc: 0.7753 - val_loss: 0.4816 - val_acc: 0.8075
Train on 186219 samples, validate on 20691 samples
Epoch 1/1
 - 88s - loss: 0.4880 - acc: 0.8076 - val_loss: 0.5199 - val_acc: 0.8046
 
'''
