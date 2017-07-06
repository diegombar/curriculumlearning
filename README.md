# Curriculum Learning for Robot Manipulation using Deep Reinforcement Learning

MSc research project on using a curriculum learning approach to train a robot arm with deep reinforcement learning.

Curriculum learning consists in breaking down a complex task into a sequence of subtasks of increasing difficulty. [Bengio et al.](http://dl.acm.org/citation.cfm?id=1553380) showed that curriculum learning approaches can shorten training time and improve generalization for supervised learning tasks, similarly to unsupervised greedy layer-wise pre-training.

The purpose of this work is to evaluate the effectiveness of curriculum learning for robot manipulation tasks, by implementing a Deep Q-learning algorithm, following the recent success of deep Q-networks at training RL agents with human-level performance at playing many Atari games ([Mnih et al.](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html))

## Files

Main files are located in [curriculumlearning/scripts/v-rep_project/](../master/scripts/v-rep_project/) :

- [training_independent_joints.py](../master/scripts/v-rep_project/training_independent_joints.py) implements the Deep Q-learning algorithm

- [robotenv.py](../master/scripts/v-rep_project/robotenv.py) implements an interface to communicate with [V-REP simulator](http://www.coppeliarobotics.com/index.html) based on [OpenAI Gym](https://gym.openai.com/docs) environments

- [MicoRobot.ttt](../master/scripts/v-rep_project/MicoRobot.ttt) is a V-REP scene file containing a model of the Mico Robot Arm

## Links

[Kinova Mico 6 DOF Robot Arm specifications](http://www.kinovarobotics.com/wp-content/uploads/2015/02/Kinova-Specs-MICO2-6DOF-Web-170512-1.pdf)

