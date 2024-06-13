import numpy as np


class ReplayBuffer():
    ''' store experience tuples (state, action, reward, next state, done) 
    and allow the agent to sample from this memory to learn more efficiently
    '''
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size  # max buffer size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)


    def store_transition(self, state, action, reward, next_state, done):
        ''' store a transition (state, action, reward, next state, done) in the buffer
        '''
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1
    

    def sample_buffer(self, batch_size):
        ''' sample a batch of transitions from the buffer
        '''
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)  # randomely select a batch of indices from the buffer

        # extract the sampled transitions based on the selected indices
        states = self.state_memory[batch]
        next_state = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_state, dones 