from gym_torcs import TorcsEnv
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam

img_dim = [64, 64, 3]
action_dim = 1
steps = 1000
batch_size = 32
epochs = 100

def get_teacher_action(ob):
    steer = ob.angle * 10 / np.pi
    steer -= ob.trackPos * 0.10
    return np.array([steer])

def img_reshape(input_img):
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img.astype(np.float32) / 255.0

images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]), dtype=np.float32)
actions_all = np.zeros((0, action_dim))
rewards_all = np.zeros((0,))

img_list = []
action_list = []
reward_list = []

env = TorcsEnv(vision=True, throttle=False)
ob = env.reset(relaunch=True)

print('Collecting data...')
for i in range(steps):
    if i == 0:
        act = np.array([0.0])
    else:
        act = get_teacher_action(ob)

    if i % 100 == 0:
        print(i)
    ob, reward, done, _ = env.step(act)
    img_list.append(ob.img)
    action_list.append(act)
    reward_list.append(np.array([reward]))

env.end()

print('Packing data into arrays...')
for img, act, rew in zip(img_list, action_list, reward_list):
    images_all = np.concatenate([images_all, img_reshape(img)], axis=0)
    actions_all = np.concatenate([actions_all, np.reshape(act, [1, action_dim])], axis=0)
    rewards_all = np.concatenate([rewards_all, rew], axis=0)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=img_dim),
    Activation('relu'),
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(action_dim),
    Activation('tanh')
])

model.compile(
    loss='mean_squared_error',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['mean_squared_error']
)

model.fit(images_all, actions_all, batch_size=batch_size, epochs=epochs, shuffle=True)

output_file = open('results.txt', 'w')

dagger_itr = 5
for itr in range(dagger_itr):
    ob_list = []
    env = TorcsEnv(vision=True, throttle=False)
    ob = env.reset(relaunch=True)
    reward_sum = 0.0

    for i in range(steps):
        act = model.predict(img_reshape(ob.img), verbose=0)
        ob, reward, done, _ = env.step(act[0])
        if done:
            break
        ob_list.append(ob)
        reward_sum += reward
        print(i, reward, reward_sum, done, str(act[0]))

    print('Episode done ', itr, i, reward_sum)
    output_file.write('Number of Steps: %02d\t Reward: %0.04f\n' % (i, reward_sum))
    env.end()

    if i == (steps - 1):
        break

    for ob in ob_list:
        images_all = np.concatenate([images_all, img_reshape(ob.img)], axis=0)
        actions_all = np.concatenate([actions_all, np.reshape(get_teacher_action(ob), [1, action_dim])], axis=0)

    model.fit(images_all, actions_all, batch_size=batch_size, epochs=epochs, shuffle=True)

output_file.close()
