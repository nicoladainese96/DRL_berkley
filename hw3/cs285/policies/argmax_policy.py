import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        #print("ArgMaxPolicy.get_action.observation.shape: ", observation.shape)
        Q = self.critic.qa_values(observation)
        action = np.argmax(Q, axis=-1) # some dim/axis needed?
        #print("action.shape: ", action.shape)
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        return action.squeeze()