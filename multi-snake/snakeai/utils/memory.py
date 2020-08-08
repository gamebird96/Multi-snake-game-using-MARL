import collections
import random

import numpy as np


class ExperienceReplay(object):
    """ Represents the experience replay memory that can be randomly sampled. """

    def __init__(self, input_shape, num_actions, memory_size=100):
        """
        Create a new instance of experience replay memory.

        Args:
            input_shape: the shape of the agent state.
            num_actions: the number of actions allowed in the environment.
            memory_size: memory size limit (-1 for unlimited).
        """
        self.memory = collections.deque()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.memory_size = memory_size

    def reset(self):
        """ Erase the experience replay memory. """
        self.memory = collections.deque()

    def remember(self, state, action, reward, state_next, is_episode_end):
        """
        Store a new piece of experience into the replay memory.

        Args:
            state: state observed at the previous step.
            action: action taken at the previous step.
            reward: reward received at the beginning of the current step.
            state_next: state observed at the current step.
            is_episode_end: whether the episode has ended with the current step.
        """
        memory_item = np.concatenate([
            state.flatten(),
            np.array(action).flatten(),
            np.array(reward).flatten(),
            state_next.flatten(),
            1 * np.array(is_episode_end).flatten()
        ])
        self.memory.append(memory_item)
        if 0 < self.memory_size < len(self.memory):
            self.memory.popleft()

    def multi_remember(self, state, action_1, action_2, reward, state_next, is_episode_end):
        """
        Store a new piece of experience into the replay memory.

        Args:
            state: state observed at the previous step.
            action: action taken at the previous step.
            reward: reward received at the beginning of the current step.
            state_next: state observed at the current step.
            is_episode_end: whether the episode has ended with the current step.
        """
        memory_item = np.concatenate([
            state.flatten(),
            np.array(action_1).flatten(),
            np.array(action_2).flatten(),
            np.array(reward).flatten(),
            state_next.flatten(),
            1 * np.array(is_episode_end).flatten()
        ])
        self.memory.append(memory_item)
        if 0 < self.memory_size < len(self.memory):
            self.memory.popleft()

    def get_batch(self, model, batch_size, discount_factor=0.9):
        """ Sample a batch from experience replay. """

        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions = experience[:, input_dim]
        rewards = experience[:, input_dim + 1]
        states_next = experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = experience[:, 2 * input_dim + 2]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions = np.cast['int'](actions)
        rewards = rewards.repeat(self.num_actions).reshape((batch_size, self.num_actions))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions).reshape((batch_size, self.num_actions))

        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = model.predict(X)
        Q_next = np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions))

        delta = np.zeros((batch_size, self.num_actions))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        return states, targets

    def get_batch_fixeddqn(self, dqn_model, target_model, batch_size, discount_factor=0.9):

        """ Sample a batch from experience replay. """

        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions = experience[:, input_dim]
        rewards = experience[:, input_dim + 1]
        states_next = experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = experience[:, 2 * input_dim + 2]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions = np.cast['int'](actions)
        rewards = rewards.repeat(self.num_actions).reshape((batch_size, self.num_actions))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions).reshape((batch_size, self.num_actions))

        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = target_model.predict(X)
        Q_next = np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions))

        delta = np.zeros((batch_size, self.num_actions))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        return states, targets

    def get_multi_batch(self, model, batch_size, discount_factor=0.9):
        """ Sample a batch from experience replay. """
        if len(self.memory) < 4: return False
        #print('Fuck')
        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions_1 = experience[:, input_dim]
        actions_2 = experience[:, input_dim + 1]
        rewards = experience[:, input_dim + 2]
        states_next = experience[:, input_dim + 3:2 * input_dim + 3]
        episode_ends = experience[:, 2 * input_dim + 3]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions_1 = np.cast['int'](actions_1)
        actions_2 = np.cast['int'](actions_2)
        #print(actions_1, actions_2)
        rewards = rewards.repeat(self.num_actions**2).reshape((batch_size, self.num_actions**2))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions**2).reshape((batch_size, self.num_actions**2))
        #print('S:', states[0], '\nA1:', actions_1[0], '\nA2:', actions_2[0], '\nR:',rewards[0],  states_next[0])
        #print('-------------------------------------------------------------------')
        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = model.predict(X)
        #print(y[batch_size:])
        #print('--------------')
        #print('Y shape', np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions)))
        #print('Gaaaaaa-------------------')
        #print(y[batch_size:].reshape((-1,self.num_actions, self.num_actions)))
        #print('bbbbbbbbbbbbbb[========================]')
        #print(np.max(np.min(y[batch_size:].reshape((-1,self.num_actions, self.num_actions)), axis=2), axis=1).repeat(self.num_actions**2).reshape((batch_size, self.num_actions**2)))
        Q_next = np.max(np.min(y[batch_size:].reshape((-1,self.num_actions, self.num_actions)), axis=2), axis=1).repeat(self.num_actions**2).reshape((batch_size, self.num_actions**2))

        delta = np.zeros((batch_size, self.num_actions*self.num_actions))
        delta[np.arange(batch_size), self.num_actions*actions_1 + actions_2] = 1
        #print(delta)

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        #p=0//0
        return states, targets

    def get_batch_doubledqn(self, model, batch_size, discount_factor=0.9):

        """ Sample a batch from experience replay. """

        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions = experience[:, input_dim]
        rewards = experience[:, input_dim + 1]
        states_next = experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = experience[:, 2 * input_dim + 2]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions = np.cast['int'](actions)
        rewards = rewards.repeat(self.num_actions).reshape((batch_size, self.num_actions))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions).reshape((batch_size, self.num_actions))

        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = model.predict(X)
        Q_next = np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions))

        delta = np.zeros((batch_size, self.num_actions))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        return states, targets
