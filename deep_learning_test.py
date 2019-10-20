#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# 28x28 images of hand written digits 0-9
mnist = tf.keras.datasets.mnist

#unpack dataset

(x_train, y_train), (x_test, y_test) =tf.keras.datasets.mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test, axis=1)

# Sequential model, 2 types of models
model = tf.keras.models.Sequential()
# 1st. layer - input layer
model.add(tf.keras.layers.Flatten())
# hidden layer - Dense layer - 128 neurons in the layer rectiffier linear
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# 2 hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# output layer 10 numbers, so 10
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#parameters, optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
model.fit(x_train, y_train, epochs=5)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

import matplotlib.pyplot as plt

#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()
#print(x_train[0])

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([x_test])

# print(predictions)

print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()
