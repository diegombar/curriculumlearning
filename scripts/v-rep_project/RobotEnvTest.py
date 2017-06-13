#!/usr/bin/env python3
import time

from micoArmContinuous import RobotEnv

with RobotEnv() as robot_env:
	print('\nstate:')
	print(robot_env.state)
	for action in range(0,18):
		state, reward, done = robot_env.step(action)
		print('\nstate:')
		print(state)
		print('\nreward:')
		print(reward)
		print('\ndone:')
		print(done)
		time.sleep(1)
	robot_env.reset()
	print('simulation RESET')
	for action in range(0,9):
		state, reward, done = robot_env.step(action)
		print('\nstate:')
		print(state)
		print('\nreward:')
		print(reward)
		print('\ndone:')
		print(done)
		time.sleep(1)

	time.sleep(10)