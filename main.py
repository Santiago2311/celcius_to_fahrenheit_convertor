"""
This program makes a neuronal network that convert celcius to fahrenheit   
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Values used to train the model
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Model with one layer
# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo = tf.keras.Sequential([capa])

# Model with more than one layer
hidden1 = tf.keras.layers.Dense(units=3, input_shape=[1])
hidden2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([hidden1, hidden2, output])

# Properties to train the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),  # .Adam(learning rate)
    loss='mean_squared_error'
)

# Trainig process of the model
print('Starting training...')

history1 = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)

print('Model trained')

# Graph of the trainig process
plt.xlabel('# Epochs')
plt.ylabel('Loss magnitud')
plt.plot(history1.history['loss'])

# Creation of a prediction to review the effectiveness of the model
print('Lets make a prediction!')

result = model.predict([100.0])

print(result)

# Shows the internal variables of the model
print('Internal model variables')
# print(capa.get_weights())
print(hidden1.get_weights())
print(hidden2.get_weights())
print(output.get_weights())
