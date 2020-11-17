import numpy as np
import gym

# from tango_connection import TangoConnection

class LaserTrajectoryControlEnv(gym.Env):

    def __init__(self, tango, **kwargs):
        self.init_rewards = []
        self.done = False
        self.current_length = 0
        self.__name__ = 'LaserTrajectoryControlEnv'
        
        self.curr_episode = -1
        self.TOTAL_COUNTER = -1
        self.rewards = []
        self.actions = []
        self.states = []
        self.dones = []
        self.initial_conditions = []
        
        self.max_length = 25
        self.max_steps = 10

        #
        self.tango = tango

        # some information from tango
        self.system = self.tango.system

        self.state_size = self.tango.state_size
        self.action_size = self.tango.action_size

        self.init_state = self.tango.init_state
        self.init_intensity = self.tango.init_intensity

        # scaling factor definition
        if 'half_range' in kwargs:
            self.half_range = kwargs.get('half_range')
        else:
            self.half_range = 3000
            if self.system == 'eos':
                self.half_range = 30000  # 30000

        self.state_range = self.get_range()
        self.state_scale = 2 * self.half_range

        # target intensity
        if 'target_intensity' in kwargs:
            self.target_intensity = kwargs.get('target_intensity')
        else:
            self.target_intensity = self.init_intensity

        # state, intensity and reward definition
        self.init_state_norm = self.scale(self.init_state)
        self.init_intensity_norm = self.get_intensity()
        self.state = self.init_state_norm.copy()
        self.intensity = self.init_intensity_norm.copy()
        self.reward = self.get_reward()

        ## max action allowed
        if 'max_action' in kwargs:
            max_action = kwargs.get('max_action')
        else:
            max_action = 500
            # bigger max_action... evalueate if the size is correct!
            # max_action = 6000 # 3000
            if self.system == 'eos':
                max_action = 5000  # 2500  # 5000
        
        self.max_action = max_action/self.state_scale
        
        # observation space definition
        self.observation_space = gym.spaces.Box(low=0.0, #+ self.max_action,
                                         high=1.0, #- self.max_action,
                                         shape=(self.state_size,),
                                         dtype=np.float64)

        # action spacec definition
        self.action_space = gym.spaces.Box(low=-self.max_action,
                                          high=self.max_action,
                                          shape=(self.action_size,),
                                          dtype=np.float64)
                                          
        self.test = False

    def get_range(self):
        # define the available state space
        state_range = np.c_[self.init_state - self.half_range, self.init_state + self.half_range]
        return state_range

    def scale(self, state):
        # scales the state from state_range values to [0, 1]
        state_scaled = (state - self.state_range[:, 0]) / self.state_scale
        return state_scaled

    def descale(self, state):
        # descales the state from [0, 1] to state_range values
        state_descaled = state * self.state_scale + self.state_range[:, 0]
        return state_descaled

    def set_state(self, state):
        # writes descaled state
        state_descaled = self.descale(state)
        self.tango.set_state(state_descaled)

    def get_state(self):
        # read scaled state
        state = self.tango.get_state()
        state_scaled = self.scale(state)
        return state_scaled

    def norm_intensity(self, intensity):
        # normalize the intensity with respect to target_intensity
        intensity_norm = intensity/self.target_intensity
        return intensity_norm

    def get_intensity(self):
        # read normalized intensity
        intensity = self.tango.get_intensity()
        intensity_norm = self.norm_intensity(intensity)
        return intensity_norm

    def step(self, action):
        # step method
        self.current_length += 1
        state, reward = self.take_action(action)
        
        intensity = self.get_intensity()
        if intensity > 0.95:
            self.done = True
            
        #elif self.current_length >= self.max_length:
        elif self.current_length >= self.max_steps:
            self.done = True
        self.add_trajectory_data(state=state, action=action, reward=reward, done=self.done)
        
        print('step', self.current_length,'state ', state, 'a ', action, 'r ', reward)
        # self.rewards[self.curr_episode].append(reward)
        
        return state, reward, self.done, {}

    def take_action(self, action):
        # initial value: action /= 12 (maybe too small)
        # action /= 12
        # take action method
        new_state = self.state + action

        # state must remain in [0, 1]
        if any(np.squeeze(new_state) < 0.0) or any(np.squeeze(new_state) > 1.0):
            new_state = np.clip(new_state, 0.0, 1.0)
            # print('WARNING: state boundaries!')

        # set new state to the machine
        self.set_state(new_state)
        state = self.get_state()
        self.state = state

        # get new intensity from the machine
        intensity = self.get_intensity()
        self.intensity = intensity

        # reward calculation
        reward = self.get_reward()
        self.reward = reward

        return state, reward

    def get_reward(self):
        # You can change reward function, but it should depend on intensity
        # e.g. next line
        # reward = -(1 - self.intensity / self.target_intensity)
        reward = -(1 - self.intensity / 1.0)

        # reward = self.intensity
        return reward

    def reset(self):
        # reset method
        
        self.done = False
        self.current_length = 0
        
        # self.curr_episode += 1
        # self.rewards.append([])
        
        bad_init = True
        while bad_init:
            new_state = self.observation_space.sample()

            self.set_state(new_state)
            state = self.get_state()
            self.state = state
             
            intensity = self.get_intensity()
            self.intensity = intensity
            self.init_rewards.append(-(1 - self.intensity / 1.0))
            
            bad_init = False
        
        self.curr_episode += 1
        self.rewards.append([])
        self.actions.append([])
        self.states.append([])
        self.dones.append([])
        # self.add_trajectory_data(state=state, action=action, reward=reward, done=done) 
        self.states[self.curr_episode].append(state)
        
        return state
    
    def add_trajectory_data(self, state, action, reward, done):
        self.rewards[self.curr_episode].append(reward)
        self.actions[self.curr_episode].append(action)
        self.states[self.curr_episode].append(state)
        self.dones[self.curr_episode].append(done) 

    def seed(self, seed=None):
        # seed method
        np.random.seed(seed)

    def render(self, mode='human'):
        # render method
        print('ERROR\nnot yet implemented!')
        pass


if __name__ == '__main__':
    
    # fel
    '''
    # system = 'eos'
    system = 'fel2'
    path = '/home/niky/FERMI/2020_10_06/configuration/'
    conf_file = 'conf_'+system+'.json'
    
    filename = path+conf_file
    tng = TangoConnection(conf_file=filename)
    env = LaserTrajectoryControlEnv(tng)
    #'''
    
