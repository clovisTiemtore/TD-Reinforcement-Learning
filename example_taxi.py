import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    # env.render()

    for i in range(4):
        env.reset()
        time.sleep(2)
