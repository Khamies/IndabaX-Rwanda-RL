import math
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.style
import seaborn as sns

from keras.models import Sequential
from keras import layers
from keras  import optimizers
from keras import losses

matplotlib.style.use('ggplot')


##########################################################################################################

# Author: Waleed Daud
# Github: waleed-daud
# Twitter: @waleeddaud
# Email: waleed.daud@outlook.com

##############################################################################################################


# Brain Settings:

BATCH_SIZE_BASELINE = 50  # calculate average reward over these many episodes
H = 64                   # hidden layer size
LEARNING_RATE=0.001


# policy Settings:
  
MAX_EPSILON = 1
MIN_EPSILON = 0.01 # stay a bit curious even when getting old

# Agent Settings

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99 # discount factor
LAMBDA = 0.0001    # speed of decay




class Brain:
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.params = {}
        self.lr = LEARNING_RATE
        self.model= self._create()
        # self.model.load_weights("cartpole-basic.h5")
        
    def _create(self):
        
        model = Sequential()
        model.add(layers.Dense(units=64, activation='relu', input_dim= self.state_size))
        model.add(layers.Dense(units= self.action_size, activation='linear'))

       
        W1=model.get_weights()[0]
        b1=model.get_weights()[1]
        W2=model.get_weights()[2]
        b2=model.get_weights()[3]
        
        self.params = dict(W1=W1, b1=b1, W2=W2, b2=b2)            
        
       
        optimizer = optimizers.adam(lr=self.lr)
        model.compile(loss='mse', optimizer=optimizer)
        

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64,epochs=epoch, verbose=verbose)
        

    def predict(self, s):
        #print(s.shape)
        return self.model.predict(s,batch_size=64)

    
####################################################################################
# Memory

class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add_sample(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def get_sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
    
####################################################################################

# Policy 
  

class Policy:

    epsilon = MAX_EPSILON
    
    def __init__(self, ACTION_COUNT):
        self.ACTION_COUNT = ACTION_COUNT
        pass
    
    def get_action(self,s,brain):
        
        if random.random() < self.epsilon:
            return random.randint(0, self.ACTION_COUNT-1)
        else:
            s=np.reshape(s,newshape=(1,s.shape[0]))
            return np.argmax(brain.predict(s)) 
    
    def decay_epsilon(self,steps):
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)
        return self.epsilon

    
#######################################################################################


class Agent:
    steps = 0

    def __init__(self, env):
        self.env = env
        self.STATE_COUNT  = self.env.observation_space.shape[0]
        self.ACTION_COUNT = self.env.action_space.n
    
        self.brain = Brain(self.STATE_COUNT,self.ACTION_COUNT)
        self.memory = Memory(MEMORY_CAPACITY)
        self.policy = Policy(self.ACTION_COUNT)
        
    def act(self, s):
        action=self.policy.get_action(s,self.brain)
        return action

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add_sample(sample)    
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.policy.decay_epsilon(self.steps)

    def replay(self):    
        batch = self.memory.get_sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.STATE_COUNT)

        
        states = np.array([ o[0] for o in batch ], dtype=np.float32)
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch ], dtype=np.float32)

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = np.zeros((batchLen, self.STATE_COUNT)).astype(np.float32)
        y = np.zeros((batchLen, self.ACTION_COUNT)).astype(np.float32)
        
        for i in range(batchLen):
            s, a, r, s_ = batch[i]
            
            t = p[i]
            if s_ is None:
                t[a] = r
                
            else:
                t[a] = r + GAMMA * np.amax(p_[i])      # calculate the target: r+ Gamma*max Q(s',a')

            x[i] = s
            y[i] = t

        self.brain.train(x, y)
        

##############################################################################################################



def run(agent, env, train = True):
    s = env.reset()
    R = 0 
    episode_length_counter=1


    while True:     
        if !train:
            env.render()


        a = agent.act(s.astype(np.float32))
        #print(a)

        s_, r, done, info = env.step(a)

        episode_length_counter+=1

        if done: # terminal state
            s_ = None
        
        if train:
            agent.observe((s, a, r, s_))
            agent.replay()            

        s = s_
        R += r

        if done:
            env.close()
            return R,episode_length_counter
        
        
###################################################################################################

# Helper tools

# Metrics

class Metric:
    
    def __init__(self,number_of_episodes):
        
        self.number_of_episodes = number_of_episodes
        self.G=[]                                           # Save the sum of rewards per episode in a list. G = [ G1,G2,G3,............]
        self.mean_G_all=[]                                  # save the mean of the episode so far in a list. mean_G = [G1, (G1+G2)/2 , (G1+G2+G3)/3, (G1+G2+G3+G4)/4, ........]
        self.episodes_length=np.zeros((self.number_of_episodes,1)) 
        
        
        self.episode_states=[]    
        self.episode_rewards=[]
        self.episode_actions=[]

        pass
    
    def reset(self):
        self.episode_states=[]    
        self.episode_rewards=[]
        self.episode_actions=[]

        return 0
        
        
    def add (self, R,episode_number,episode_length_counter):
        self.R = R
        self.episode_number = episode_number
        self.episode_length_counter = episode_length_counter

        self.G.append(self.R)
        self.episodes_length[self.episode_number]=self.episode_length_counter
        
        return 0
    
    
    def show(self):
                        
        print("==========================================")
        print("Episode: ", self.episode_number+1)
        print("Rewards: ", self.R)
        print("Max reward so far: ", max(self.G))
        # Mean reward
        mean_G = np.divide(sum(self.G), self.episode_number+1)
        self.mean_G_all.append(mean_G)
        print("Mean Reward", mean_G)
        
        return 0