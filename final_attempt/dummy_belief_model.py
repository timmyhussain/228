import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from helper import Grid, get_error
import matplotlib.pyplot as plt


a = lambda : np.random.randint(0,2, size=(1,8))
b = lambda x : np.hstack([np.array(10, np.int32, ndmin=2), 10*(1-x)])
# b = lambda x: 10*(1-x)
obstacles = np.load("map1-new/obstacles.npy", allow_pickle=True)
grid = Grid(10, 10, 10, -65.32, -53.20, obstacles)
np.random.seed(0)
belief, _, goal = grid.reset()

# print(belief)

action_values = [0]*9
direct_action_rewards = [0]*9
future_state_rewards = [0]*9
gamma = 0.9

errors = []
# errors_actions = []

# for a in range(9):
#     np.random.seed(0)
#     belief, _, goal = grid.reset()
#     errors_actions = []
#     for i in range(10):
#         errors_actions.append(get_error(belief, *goal))
#         new_belief, _, _, reward, _, _ = grid.step(belief, a)
#         belief = new_belief.copy()
#         # action_values[a] = reward + gamma*grid.compute_reward(new_belief, 0, 0)[0]
#         # direct_action_rewards[a] = reward
#         # future_state_rewards[a] = grid.compute_reward(new_belief, 0, 0)[0]
#     errors.append(errors_actions)


# for a in range(9):
#     # plt.plot(np.diff(errors[a]), label=a)
#     plt.plot(errors[a], label=a)

# plt.legend()
# plt.show()

beliefs = []
action_values = []
goals = []
for i in range(100):
    np.random.seed(i)
    belief, _, goal = grid.reset()
    errors_actions = []
    action_value = [0]*9
    for a in range(9):
        # errors_actions.append(get_error(belief, *goal))
        new_belief, _, _, reward, _, _ = grid.step(belief, a)
        action_value[a] = reward + gamma*grid.compute_reward(new_belief, 0, 0)[0]
        # direct_action_rewards[a] = reward
        # future_state_rewards[a] = grid.compute_reward(new_belief, 0, 0)[0]
        # errors_actions.append(get_error(new_belief, *goal))
        # errors.append(errors_actions)
    beliefs.append(belief.flatten())
    goals.append(goal)
    action_values.append(action_value)

# print(errors_actions)
# print("action rewards: ", direct_action_rewards)
# print("future rewards: ", future_state_rewards)
# print("action values: ", action_values)


beliefs = np.vstack(beliefs)
goals = np.vstack(goals)
print(goals.shape)
action_values = np.vstack(action_values)

# print(inputs[:20, :])
# print(outputs) 
# time.sleep(10)
belief_input = keras.Input(shape=(144,), name="beliefs")
goal_input = keras.Input(shape=(2,), name="goals")

belief_features = layers.Dense(720, activation="relu")(belief_input)
goal_features = layers.Embedding(input_dim=(12), output_dim=1, input_length=2)(goal_input)
goal_features = layers.Dense(10, activation="relu")(goal_features)
goal_features = layers.Flatten()(goal_features)
joint_features = layers.Concatenate()([belief_features, goal_features])
# goal_features = layers.Dense(10, activation="relu")(goal_input)
# print(goal_features)
# obstacle_features = layers.Dense(40, activation="relu")(obstacle_features)

# joint_output = layers.Dense(9, activation="linear", name="values")(goal_features)
joint_output = layers.Dense(9, activation="linear", name="values")(joint_features)
# obstacle_output = layers.Lambda(keras.backend.abs)(obstacle_output)



model = keras.Model(
    inputs = [belief_input, goal_input],
    outputs=[joint_output])

optimizer = keras.optimizers.Adam(learning_rate = 0.01)

# loss = {"belief": keras.losses.MeanSquaredError()}#did ok with mse

# loss = {keras.losses.MeanSquaredError()} #did ok with mse
loss = keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss="mse")#, metrics=metrics)

for n in range(300):
    model.fit({"beliefs": beliefs, "goals": goals}, {"values": action_values})
# arr = np.random.randint(0,2, size=(1,8))

np.random.seed(0)
belief, _, goal = grid.reset()
new_belief, _, _, reward, _, _ = grid.step(belief, 8)
print(grid.goal)
print(model.predict({"beliefs": new_belief.flatten().reshape(1, -1), "goals": goal.reshape(1,2)}))

model.save("belief_model/model.h5")
# print(arr)
# print(model.predict(arr))


# model = keras.models.load_model("belief_model/model.h5")
# arr = np.random.randint(0,2, size=(1,8))
# print(arr)
# print(model.predict(arr))

