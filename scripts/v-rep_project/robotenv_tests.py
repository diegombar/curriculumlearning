#!/usr/bin/env python3
import time
from robotenv import RobotEnv
import numpy as np


def test_individual_joints(env):
    nJoints = 6
    # test individual joints
    for i in range(nJoints):
        actions = np.ones((5, 6))  # 1 corresponds to zero velocity
        actions[:, i] = 0, 1, 2, 0, 1
        # state = env.reset()
        # print('\nstate0:', state)
        for action in actions:
            state, reward, done = env.step(action)
            print('\nstate:')
            print(state)
            # print('\nreward:')
            # print(reward)
            # print('\ndone:')
            # print(done)
            time.sleep(0.1)


with RobotEnv(1) as env:
    test_individual_joints(env)
    env.reset()
