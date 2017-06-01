import numpy as np
import itertools
import scipy.misc
import matplotlib.pyplot as plt


class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class gameEnv():
    def __init__(self,partial,size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        a, state = self.reset()
        plt.imshow(a,interpolation="nearest")
        
        
    def reset(self):
        self.objects = []
        hero = gameOb((0,0),1,1,2,None,'hero')
        self.objects.append(hero)
        goal = gameOb((3,3),1,1,1,1,'goal')
        self.objects.append(goal)
        hole = gameOb((2,1),1,1,0,-1,'fire')
        self.objects.append(hole)
        hole2 = gameOb((1,2),1,1,0,-1,'fire')
        self.objects.append(hole2)
        hole3 = gameOb((3,0),1,1,0,-1,'fire')
        self.objects.append(hole3)
        # bug2 = gameOb(self.newPosition(),1,1,1,1,'goal')
        # self.objects.append(bug2)
        # bug3 = gameOb(self.newPosition(),1,1,1,1,'goal')
        # self.objects.append(bug3)
        # bug4 = gameOb(self.newPosition(),1,1,1,1,'goal')
        # self.objects`.append(bug4)
        a, state = self.renderEnv()
        self.state = state
        return a, state

    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = -0.01
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1
        if hero.x == heroX and hero.y == heroY:
            penalize = -0.01
        self.objects[0] = hero
        return penalize
    
    # randomly pick a new position
    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                return other.reward, True
        if not ended:
            return 0.0, False

    def renderEnv(self):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self.sizeY+2,self.sizeX+2,3])
        a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]
        b = scipy.misc.imresize(a[:,:,0],[84,84,1],interp='nearest')
        c = scipy.misc.imresize(a[:,:,1],[84,84,1],interp='nearest')
        d = scipy.misc.imresize(a[:,:,2],[84,84,1],interp='nearest')
        a = np.stack([b,c,d],axis=2)
        heroPos = hero.x + self.sizeX * hero.y
        return a, heroPos

    def step(self, action):
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        a, heroPos = self.renderEnv()
        return heroPos, (reward + penalty), done, a
