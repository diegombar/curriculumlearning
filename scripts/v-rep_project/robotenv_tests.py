#!/usr/bin/env python3
import time

from robotenv import RobotEnv

with RobotEnv(1) as robot_env:
	state = robot_env.reset()
	print('\nstate0:', state)
	actions = [[0,0,0,0,0,0], [1,0,0,0,0,0], [2,0,0,0,0,0]]
	for action in actions:
		state, reward, done = robot_env.step(action)
		print('\nstate:')
		print(state)
		print('\nreward:')
		print(reward)
		print('\ndone:')
		print(done)
		time.sleep(1)
	state = robot_env.reset()
	print('\nstate0:', state)
	print('simulation RESET')
	actions = [[0,0,0,0,0,0], [0,1,0,0,0,0], [0,2,0,0,0,0]]
	for action in actions:
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

	time.sleep(10)