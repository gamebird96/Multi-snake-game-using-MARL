import collections
import numpy as np

from snakeai.agent import AgentBase
from snakeai.utils.memory import ExperienceReplay


class MinimaxDeepQNetworkAgent(AgentBase):


    def __init__(self, model_1, model_2, num_last_frames=4, memory_size=1000):
        """
        Create a new DQN-based agent.

        Args:
            model_1: a compiled DQN model for snake 1.
            model_2: a compiled DQN model for snake 2.
            num_last_frames (int): the number of last frames the agent will consider.
            memory_size (int): memory size limit for experience replay (-1 for unlimited).
        """
        assert model_1.input_shape[1] == num_last_frames, 'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model_1.output_shape) == 2, 'Model output shape should be (num_samples, num_actions)'
        assert model_2.input_shape[1] == num_last_frames, 'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model_2.output_shape) == 2, 'Model output shape should be (num_samples, num_actions)'

        self.model_1 = model_1
        self.model_2 = model_2
        self.num_last_frames = num_last_frames
        self.memory_1 = ExperienceReplay((num_last_frames,) + model_1.input_shape[-2:], model_1.output_shape[-1]//3, memory_size)
        self.memory_2 = ExperienceReplay((num_last_frames,) + model_2.input_shape[-2:], model_2.output_shape[-1]//3, memory_size)
        self.frames = None

    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.frames = None

    def get_last_frames(self, observation):
        """
        Get the pixels of the last `num_last_frames` observations, the current frame being the last.

        Args:
            observation: observation at the current timestep.

        Returns:
            Observations for the last `num_last_frames` frames.
        """
        frame = observation
        if self.frames is None:
            self.frames = collections.deque([frame] * self.num_last_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.expand_dims(self.frames, 0).astype(np.float32)/16

    def train(self, env, num_episodes=1000, batch_size=50, discount_factor=0.9, checkpoint_freq=None,
              exploration_range=(1.0, 0.1), exploration_phase_size=0.5):
        """
        Train the agent to perform well in the given Snake environment.

        Args:
            env:
                an instance of Snake environment.
            num_episodes (int):
                the number of episodes to run during the training.
            batch_size (int):
                the size of the learning sample for experience replay.
            discount_factor (float):
                discount factor (gamma) for computing the value function.
            checkpoint_freq (int):
                the number of episodes after which a new model checkpoint will be created.
            exploration_range (tuple):
                a (max, min) range specifying how the exploration rate should decay over time.
            exploration_phase_size (float):
                the percentage of the training process at which
                the exploration rate should reach its minimum.
        """

        # Calculate the constant exploration decay speed for each episode.
        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (num_episodes * exploration_phase_size))
        exploration_rate = max_exploration_rate

        for episode in range(num_episodes):
            # Reset the environment for the new episode.
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss_1 = 0.0
            loss_2 = 0.0
            alive_1 = True
            alive_2 = True
            # Observe the initial state.
            state = self.get_last_frames(timestep.observation)

            while not game_over:
                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action = (np.random.randint(env.num_actions), np.random.randint(env.num_actions))
                else:
                    # Exploit: take the best known action for this state.
                    q1 = self.model_1.predict(state)
                    q2 = self.model_2.predict(state)
                    q1 = q1.reshape((env.num_actions, env.num_actions))
                    q2 = q2.reshape((env.num_actions, env.num_actions))
                    if alive_1 and alive_2:
                        action = (np.argmax(np.min(q1, axis=1)), np.argmax(np.min(q2, axis=1)))
                    elif alive_1:
                        action = (np.argmax(np.min(q1, axis=1)), np.argmin(np.max(q1, axis=0)))
                    elif alive_2:
                        action = (np.argmin(np.max(q2, axis=0)), np.argmax(np.min(q2, axis=1)))

                # Act on the environment.
                env.choose_action(action)
                timestep = env.timestep()

                # Remember a new piece of experience.
                reward_1, reward_2 = timestep.reward_1, timestep.reward_2
                state_next = self.get_last_frames(timestep.observation)
                game_over = timestep.is_episode_end

                experience_item_1 = [state, action[0], action[1], reward_1, state_next, game_over]
                experience_item_2 = [state, action[1], action[0], reward_2, state_next, game_over]
                self.memory_1.multi_remember(*experience_item_1)
                self.memory_2.multi_remember(*experience_item_2)
                state = state_next

                # Sample a random batch from experience.

                if alive_1:
                    batch = self.memory_1.get_multi_batch(
                        model=self.model_1,
                        batch_size=batch_size,
                        discount_factor=discount_factor
                    )
                    # Learn on the batch.
                    if batch:
                        inputs, targets = batch
                        loss_1 += float(self.model_1.train_on_batch(inputs, targets))
                    # Sample a random batch from experience.

                if alive_2:
                    batch = self.memory_2.get_multi_batch(
                        model=self.model_2,
                        batch_size=batch_size,
                        discount_factor=discount_factor
                    )
                    # Learn on the batch.
                    if batch:
                        inputs, targets = batch
                        loss_2 += float(self.model_2.train_on_batch(inputs, targets))

                alive_1 = timestep.alive_1
                alive_2 = timestep.alive_2


            if checkpoint_freq and (episode % checkpoint_freq) == 0:
                self.model_1.save(f'dqn-mm1-{episode:08d}.model')
                self.model_2.save(f'dqn-mm2-{episode:08d}.model')

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            summary = 'Episode {:5d}/{:5d} | Loss {:8.4f}, {:8.4f} | Exploration {:.2f} | ' + \
                      'Fruits {:2d}, {:2d} | Timesteps {:4d} | Total Reward {:4d}, {:4d}'
            print(summary.format(
                episode + 1, num_episodes, loss_1, loss_2, exploration_rate,
                env.stats.fruits_eaten_1, env.stats.fruits_eaten_2, env.stats.timesteps_survived,
                env.stats.sum_episode_rewards_1, env.stats.sum_episode_rewards_2

            ))

        self.model_1.save('dqn-mm1-final.model')
        self.model_2.save('dqn-mm2-final.model')

    def act(self, observation, reward,  alive_1=True, alive_2=True):
        """
        Choose the next action to take.

        Args:
            observation: observable state for the current timestep.
            reward: reward received at the beginning of the current timestep.

        Returns:
            The index of the action to take next.
        """
        state = self.get_last_frames(observation)
        q1 = self.model_1.predict(state).reshape(3, 3)
        q2 = self.model_2.predict(state).reshape(3, 3)

        if alive_1 and alive_2: return (np.argmax(np.min(q1, axis=1)), np.argmax(np.min(q2, axis=1)))
        elif alive_1: return (np.argmax(np.min(q1, axis=1)), np.argmin(np.max(q1, axis=0)))
        elif alive_2: return (np.argmin(np.max(q2, axis=1)), np.argmax(np.min(q2, axis=0)))
