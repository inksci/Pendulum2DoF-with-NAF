TCP_PORT = 15001

from ink_v1.env_tcp import env_server
import numpy as np
import gym

class env_2Pendulum():
    def __init__(self):
        
        self.DoF = DoF = 2
        self.state_dim = 6
        self.action_dim = DoF

        self.env1 = gym.make('Pendulum-v0')
        self.env2 = gym.make('Pendulum-v0')

        self.max_steps = 300

        # Delete them please.
        self.action_ampl = -1
        self.v_lmt = -1
        self.time_step = -1

    def reset(self):
        s1 = self.env1.reset()
        s2 = self.env2.reset()
        
        return np.array( [s1,s2] ).reshape(-1)
    def step(self, action):
        action = np.array(action).reshape(-1)
        s1, r1, d1, info1 = self.env1.step( action[:1] )
        s2, r2, d2, info2 = self.env2.step( action[1:] )

        s = np.array( [s1,s2] ).reshape(-1)

        r = (r1+r2)/2.0

        d = 0

        info = "info"

        return s, r, d, info
    
env = env_2Pendulum()

env_server( env, tcp_port=TCP_PORT )