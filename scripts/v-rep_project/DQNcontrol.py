# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import random
import tensorflow as tf
from matplotlib import pyplot as plt
import readchar
import time


# Load environment
from robotenv import RobotEnv

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

with RobotEnv() as env:
    # network
    tf.reset_default_graph()
    # feed-forward part to choose actions
    stateSize = env.observation_space_size
    nActions = env.action_space_size
    nhidden = 256
    lrate = 1E-4 #try 6E-6 
    inState = tf.placeholder(shape=[1,stateSize], dtype=tf.float32)

    # initialize params
    W0 = weight_variable([stateSize, nhidden])
    W1 = weight_variable([nhidden, nhidden])
    W2 = weight_variable([nhidden, nActions])
    b0 = bias_variable([1])
    b1 = bias_variable([1])
    b2 = bias_variable([1])

    # layers
    hidden1 = tf.nn.relu(tf.matmul(inState, W0) + b0)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W1) + b1)
    outQvalues = tf.matmul(hidden2, W2) + b2 # Q values for all actions given inState
    bestAction = tf.argmax(outQvalues, 1) # action with highest Q given inState

    # loss by taking the sum of squares difference between the target and prediction Q values.
    Qtargets = tf.placeholder(shape=[1,nActions], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(Qtargets - outQvalues))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate) #try Adam? 
    updateModel = optimizer.minimize(loss)

    # Training
    init = tf.global_variables_initializer()

    # Set learning parameters
    y = 0.99 # discount factor
    
    num_episodes = 20 # number of runs

    max_steps_per_episode = 500 # max number of actions per episode
    e_max = 1.0 # initial epsilon
    e_min = 0.01 # final epsilon
    num_e_updates = 10 # times e is decreased
    e_update_period = num_episodes // num_e_updates # num of episodes between e updates

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    message = "\nepisode: {} steps: {} total reward: {} done: {}"
    with tf.Session() as sess:
        start_time = time.time()
        sess.run(init)
        for i in range(num_episodes):
            # reset environment and get first new observation
            initialState = env.reset()
            rAll = 0
            done = False
            j = 0
            #The Q-Network
            while j < max_steps_per_episode:
                j+=1
                # pick action from the DQN, epsilon greedy
                chosenAction, allQvalues = sess.run([bestAction,outQvalues], feed_dict={inState:initialState})
                if np.random.rand(1) < e: chosenAction[0] = np.random.randint(0, nActions-1)

                # perform action and get new state and reward
                newState, r, done = env.step(chosenAction[0])

                # get Q values for new state 
                allNewQvalues = sess.run(outQvalues,feed_dict={inState:newState})
                # get max for Q target
                maxQ = np.max(allNewQvalues)
                targetQ = allQvalues # dont update Q values corresponding for actions not performed
                targetQ[0, chosenAction[0]] = r + y * maxQ #only update Q value for action performed
                #Train our network using target and predicted Q values
                sess.run([updateModel], feed_dict={inState:initialState, Qtargets:targetQ})
                rAll += r
                initialState = newState

                if done == True:
                    break

            if i % e_update_period ==0:
                # decrease epsilon
                e = e - (e_max - e_min)/num_e_updates
                print('\nepsilon:', e)
                    
            if i % 10 == 0:
                # training progres log
                logMessage = message.format(i,j,rAll,done)
                print(logMessage)
            jList.append(j)
            rList.append(rAll)
        
        
        end_time = time.time()
        print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

        print("press 'v' to start policy test visualization")
        while True:
            c = readchar.readchar()
            print('char=',c)
            if c == 'v':
                break

        #Visualization of learned policy
        #alternatively, save the weights W0, W1, W2
        initialState = env.reset()
        rAll = 0
        done = False
        j = 0
        while j < max_steps_per_episode:
            j+=1
            chosenAction = sess.run(bestAction,feed_dict={inState:initialState})
            newState, r, done = env.step(chosenAction[0])
            rAll += r
            initialState = newState
            if done == True:
                break
        print('\nsteps:', j)
        print('\nreturn:', rAll)     


# statistics

# time
elapsed_time = end_time - start_time #in seconds
print('\nTotal training time:', elapsed_time)


# total return observed in each episode
plt.figure()
plt.plot(rList, 'o')
plt.ylabel('return obtained')
plt.xlabel('episode number')

# steps performed in each episode
plt.figure()
plt.plot(jList, 'o')
plt.ylabel('number of steps performed')
plt.xlabel('episode number')

plt.show()

