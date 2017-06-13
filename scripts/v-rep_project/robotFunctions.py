class robotState():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class robotEnv():
	# initialize the environment
    def __init__(self):
    	self.action_space_size = 3 * 6 # (+Vel, -Vel, 0) for 6 joints
    	self.observation_space_size = 6
        self.action_space = range(0,self.action_space_size)
        self.observation_space = [0] * self.observation_space_size
        #initialize V-REP here?
        state = self.reset()
    
    # reset the state for each new episode
    def reset(self):
    	#reset robot position in V-REP, i.e. reset simulation (or reset scene?)
        return state

    # check if a goal state has been reached
    def checkDone(self):
    	return false

   	# execute action
    def step(self, action):
        return new_state, reward, done
