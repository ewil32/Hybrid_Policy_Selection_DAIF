import gym
import numpy as np

env = gym.make('MountainCarContinuous-v0')

env.reset()

while True:

    action = float(input())
    action = np.array([action])

    print(env.step(action))
