import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


a = lambda : np.random.randint(0,2, size=(1,8))
b = lambda x : np.hstack([np.array(10, np.int32, ndmin=2), 10*(1-x)])
# b = lambda x: 10*(1-x)

inputs = []
outputs = []
for i in range(500):
    # if i % 10 == 0:
        # np.random.seed(0)
    arr = a()
    inputs.append(arr)
    outputs.append(b(arr))

inputs = np.vstack(inputs)
outputs = np.vstack(outputs)

# print(inputs[:20, :])
# print(outputs) 
# time.sleep(10)
obstacle_input = keras.Input(shape=(8,), name="obstacles")

obstacle_features = layers.Dense(80, activation="relu")(obstacle_input)
# obstacle_features = layers.Dense(40, activation="relu")(obstacle_features)

obstacle_output = layers.Dense(9, activation="linear")(obstacle_features)
# obstacle_output = layers.Lambda(keras.backend.abs)(obstacle_output)

model = keras.Model(
    inputs = [obstacle_input],
    outputs=[obstacle_output])

optimizer = keras.optimizers.Adam(learning_rate = 0.01)

# loss = {"belief": keras.losses.MeanSquaredError()}#did ok with mse

# loss = {keras.losses.MeanSquaredError()} #did ok with mse
loss = keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss="mse")#, metrics=metrics)

for n in range(300):
    model.fit(inputs, outputs)
arr = np.random.randint(0,2, size=(1,8))

model.save("lidar_model/model.h5")
print(arr)
print(model.predict(arr))


model = keras.models.load_model("lidar_model/model.h5")
arr = np.random.randint(0,2, size=(1,8))
print(arr)
print(model.predict(arr))