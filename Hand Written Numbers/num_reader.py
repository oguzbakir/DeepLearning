import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # Flatten input layer

# 2 Hidden dense layers with 128 neurons
# Activation function is relu function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Training with "adam" optimizer
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )

# Train the model
model.fit(x_train, y_train, epochs=3)

# Calculate validation loss and validation accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Save model
# To read model use
# new_model = tf.keras.models.load_model(MODEL_NAME)
model.save('num_reader.model')

# Let's predict
predictions = model.predict([x_test])
print(np.argmax(predictions[0])) # It says first element is a 7. We can draw it to check
plt.imshow(x_test[0])
plt.show() # Yup, it's a 7

#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()