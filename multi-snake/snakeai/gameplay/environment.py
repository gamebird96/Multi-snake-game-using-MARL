import pprint
import random
import time

import numpy as np
import pandas as pd

from .entities import Snake, Field, CellType, SnakeAction, ALL_SNAKE_ACTIONS


class Environment(object):
    """
    Represents the RL environment for the Snake game that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """

    def __init__(self, config, verbose=1):
        """
        Create a new Snake RL environment.

        Args:
            config (dict): level configuration, typically found in JSON configs.
            verbose (int): verbosity level:
                0 = do not write any debug information;
                1 = write a CSV file containing the statistics for every episode;
                2 = same as 1, but also write a full log file containing the state of each timestep.
        """
        self.field = Field(level_map=config['field'])
        self.snake = None
        '''ADDITIONAL CODE'''
        self.snake_2 = None
        self.fruit = None
        self.initial_snake_length = config['initial_snake_length']
        self.rewards = config['rewards']
        self.max_step_limit = config.get('max_step_limit', 1000)
        self.is_game_over = False
        self.timestep_index = 0
        self.current_action = None
        self.alive_1 = True
        self.alive_2 = True
        self.stats = EpisodeStatistics()
        self.verbose = verbose
        self.debug_file = None
        self.stats_file = None

    def seed(self, value):
        """ Initialize the random state of the environment to make results reproducible. """
        random.seed(value)
        np.random.seed(value)

    @property
    def observation_shape(self):
        """ Get the shape of the state observed at each timestep. """
        return self.field.size, self.field.size

    @property
    def num_actions(self):
        """ Get the number of actions the agent can take. """
        return len(ALL_SNAKE_ACTIONS)

    def new_episode(self):
        """ Reset the environment and begin a new episode. """
        self.field.create_level()
        self.stats.reset()
        self.timestep_index = 0

        self.snake = Snake(self.field.find_snake_head(), length=self.initial_snake_length)
        '''ADDITIONAL CODE'''
        self.snake_2 = Snake(self.field.find_snake_head_2(), length=self.initial_snake_length)
        self.field.place_snake(self.snake)
        self.field.place_snake_2(self.snake_2)
        self.generate_fruit()
        self.current_action = None
        self.is_game_over = False
        self.alive_1 = True
        self.alive_2 = True
        self.purge_1 = False
        self.purge_2 = False
        result = TimestepResult(
            observation=self.get_observation(),
            reward_1 = 0,
            reward_2 = 0,
            alive_1 = True,
            alive_2 = True,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result

    def record_timestep_stats(self, result):
        """ Record environment statistics according to the verbosity level. """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # Write CSV header for the stats file.
        if self.verbose >= 1 and self.stats_file is None:
            self.stats_file = open(f'snake-env-{timestamp}.csv', 'w')
            stats_csv_header_line = self.stats.to_dataframe()[:0].to_csv(index=None)
            print(stats_csv_header_line, file=self.stats_file, end='', flush=True)

        # Create a blank debug log file.
        if self.verbose >= 2 and self.debug_file is None:
            self.debug_file = open(f'snake-env-{timestamp}.log', 'w')

        self.stats.record_timestep(self.current_action, result)
        self.stats.timesteps_survived = self.timestep_index

        if self.verbose >= 2:
            print(result, file=self.debug_file)

        # Log episode stats if the appropriate verbosity level is set.
        if result.is_episode_end:
            if self.verbose >= 1:
                stats_csv_line = self.stats.to_dataframe().to_csv(header=False, index=None)
                print(stats_csv_line, file=self.stats_file, end='', flush=True)
            if self.verbose >= 2:
                print(self.stats, file=self.debug_file)

    def get_observation(self):
        """ Observe the state of the environment. """
        return np.copy(self.field._cells)
    '''
    def choose_action(self, action):
        """ Choose the action that will be taken at the next timestep. """

        self.current_action = action
        if action == SnakeAction.TURN_LEFT:
            self.snake.turn_left()
            self.snake_2.turn_left()
        elif action == SnakeAction.TURN_RIGHT:
            self.snake.turn_right()
            self.snake_2.turn_right()

    '''
    def choose_action(self, action):

        """ Choose the action that will be taken at the next timestep. """
        action_1 = action[0]
        action_2 = action[1]

        self.current_action = action
        if action_1 == SnakeAction.TURN_LEFT and self.alive_1:
            self.snake.turn_left()

        elif action_1 == SnakeAction.TURN_RIGHT and self.alive_1:
            self.snake.turn_right()

        if action_2 == SnakeAction.TURN_LEFT and self.alive_2:
            self.snake_2.turn_left()

        elif action_2 == SnakeAction.TURN_RIGHT and self.alive_2:
            self.snake_2.turn_right()


    def timestep(self):
        """ Execute the timestep and return the new observable state. """

        self.timestep_index += 1
        reward_1 = 0
        reward_2 = 0

        old_head = self.snake.head
        old_tail = self.snake.tail

        '''ADDITIONAL CODE'''
        old_head_2 = self.snake_2.head
        old_tail_2 = self.snake_2.tail

        # Are we about to eat the fruit?
        if self.alive_1:
            if self.snake.peek_next_move() == self.fruit:
                self.snake.grow()
                self.generate_fruit()
                old_tail = None
                reward_1 += self.rewards['ate_fruit'] * self.snake.length + 1 - int(self.alive_2)
                if self.alive_2: reward_2 -= reward_1
                self.stats.fruits_eaten_1 += 1

            # If not, just move forward.
            else:
                self.snake.move()
                reward_1 += self.rewards['timestep']
            '''ADDITIONAL CODE'''
        # Are we about to eat the- fruit?
        if self.alive_2:
            if self.snake_2.peek_next_move() == self.fruit:
                self.snake_2.grow()
                self.generate_fruit()
                old_tail_2 = None
                reward_2 += self.rewards['ate_fruit'] * self.snake_2.length + 1 - int(self.alive_1)
                if self.alive_1: reward_1 -= reward_2
                self.stats.fruits_eaten_2 += 1

            # If not, just move forward.
            else:
                self.snake_2.move()
                reward_2 += self.rewards['timestep']

        if self.alive_1: self.field.update_snake_footprint(old_head, old_tail, self.snake.head)
        if self.alive_2: self.field.update_snake_footprint_2(old_head_2, old_tail_2, self.snake_2.head)
        # Hit a wall or own body?


        if self.alive_1 and self.has_hit_wall():
            self.stats.termination_reason = 'hit_wall'
            self.field[self.snake.head] = CellType.SNAKE_HEAD
            reward_1 = self.rewards['died']
            if self.alive_2: reward_2 = -reward_1
            self.alive_1 = False

        if self.alive_1 and self.has_hit_own_body():
            self.stats.termination_reason = 'hit_own_body'
            self.field[self.snake.head] = CellType.SNAKE_HEAD
            reward_1 = self.rewards['died']
            if self.alive_2: reward_2 = -reward_1
            self.alive_1 = False

        if self.alive_1 and self.alive_2 and self.collision():

            self.stats.termination_reason = 'hit_body'
            self.field[self.snake.head] = CellType.SNAKE_HEAD
            reward_1 = self.rewards['died']
            if self.alive_2: reward_2 = -reward_1
            self.alive_1 = False

        if self.alive_2 and self.has_hit_wall_2():
            self.stats.termination_reason = 'hit_wall'
            self.field[self.snake_2.head] = CellType.SNAKE_HEAD_2
            reward_2 = self.rewards['died']
            if self.alive_1: reward_1 = -reward_2
            self.alive_2 = False

        if self.alive_2 and self.has_hit_own_body_2():
            self.stats.termination_reason = 'hit_own_body'
            self.field[self.snake_2.head] = CellType.SNAKE_HEAD_2
            reward_2 = self.rewards['died']
            if self.alive_1: reward_1 = -reward_2

            self.alive_2 = False
        if self.alive_1 and self.alive_2 and self.collision_2():

            self.stats.termination_reason = 'hit_body'
            self.field[self.snake.head] = CellType.SNAKE_HEAD_2
            reward_2 = self.rewards['died']
            if self.alive_1: reward_1 = -reward_2
            self.alive_2 = False

        if self.head_on_collision():
            turn = np.random.choice(2)
            if turn == 0:
                reward_1 = 1
                reward_2 = -1
            else:
                reward_1 = -1
                reward_2 = 1

            self.alive_1 = False
            self.alive_2 = False

        self.is_game_over = not self.alive_1 and not self.alive_2
        if not self.alive_1 and not self.purge_1:
            self.purge_1 = True
            for  i in range(1,len(self.snake.body)):
                self.field[self.snake.body[i]] = CellType.EMPTY

        if not self.alive_2 and not self.purge_2:
            self.purge_2 = True
            for  i in range(1,len(self.snake_2.body)):
                self.field[self.snake_2.body[i]] = CellType.EMPTY

        # Exceeded the limit of moves?
        if self.timestep_index >= self.max_step_limit:
            self.is_game_over = True
            self.stats.termination_reason = 'timestep_limit_exceeded'

        result = TimestepResult(
            observation=self.get_observation(),
            reward_1 = reward_1,
            reward_2 = reward_2,
            alive_1 = self.alive_1,
            alive_2 = self.alive_2,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result

    def generate_fruit(self, position=None):
        """ Generate a new fruit at a random unoccupied cell. """
        if position is None:
            position = self.field.get_random_empty_cell()
        self.field[position] = CellType.FRUIT
        self.fruit = position

    def has_hit_wall(self):
        """ True if the snake has hit a wall, False otherwise. """
        return self.field[self.snake.head] == CellType.WALL

    def has_hit_wall_2(self):
        """ True if the snake has hit a wall, False otherwise. """
        return self.field[self.snake_2.head] == CellType.WALL

    def has_hit_own_body(self):
        """ True if the snake has hit its own body, False otherwise. """
        return self.field[self.snake.head] == CellType.SNAKE_BODY

    def has_hit_own_body_2(self):
        """ True if the snake has hit its own body, False otherwise. """
        return self.field[self.snake_2.head] == CellType.SNAKE_BODY_2

    def collision(self):
        """ True if snake crashed into snake_2 """
        #return self.field[self.snake.head] == CellType.SNAKE_BODY_2
        return self.snake.head in self.snake_2.body

    def collision_2(self):
        """ True if snake_2 crashed into snake """
        #return self.field[self.snake_2.head] == CellType.SNAKE_BODY
        return self.snake_2.head in self.snake.body

    def head_on_collision(self):
        """ True if head on collision occurs """
        return self.collision() and self.collision_2()


    def collision_check(self):
        """ True if the snake is still alive, False otherwise. """
        #return not self.has_hit_wall() and not self.has_hit_own_body()

        ''' ADDITIONAL CODE'''
        return not self.has_hit_wall() and not self.has_hit_own_body() and not self.collision() and not self.has_hit_wall_2() and not self.has_hit_own_body_2() and not self.collision_2() and not self.head_on_collision()


class TimestepResult(object):
    """ Represents the information provided to the agent after each timestep. """

    def __init__(self, observation, reward_1, reward_2, alive_1, alive_2, is_episode_end):
        self.observation = observation
        self.reward_1 = reward_1
        self.reward_2 = reward_2
        self.alive_1 = alive_1
        self.alive_2 = alive_2
        self.is_episode_end = is_episode_end

    def __str__(self):
        field_map = '\n'.join([
            ''.join(str(cell) for cell in row)
            for row in self.observation
        ])
        return f'{field_map}\nR = {self.reward_1}   end={self.is_episode_end}\n'


class EpisodeStatistics(object):
    """ Represents the summary of the agent's performance during the episode. """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Forget all previous statistics and prepare for a new episode. """
        self.timesteps_survived = 0
        self.sum_episode_rewards_1 = 0
        self.sum_episode_rewards_2 = 0
        self.fruits_eaten_1 = 0
        self.fruits_eaten_2 = 0
        self.termination_reason = None
        #self.action_counter = {
        #    action: 0
        #    for action in ALL_SNAKE_ACTIONS
        #}
        self.action_counter = dict()
        for i in ALL_SNAKE_ACTIONS:
            for j in ALL_SNAKE_ACTIONS:
                self.action_counter[(i,j)] = 0

    def record_timestep(self, action, result):
        """ Update the stats based on the current timestep results. """
        self.sum_episode_rewards_1 += result.reward_1
        self.sum_episode_rewards_2 += result.reward_2
        if action is not None:
            self.action_counter[action] += 1

    def flatten(self):
        """ Format all episode statistics as a flat object. """
        flat_stats = {
            'timesteps_survived': self.timesteps_survived,
            'sum_episode_rewards_1': self.sum_episode_rewards_1,
            'sum_episode_rewards_2': self.sum_episode_rewards_2,
            'mean_reward_1': self.sum_episode_rewards_1 / self.timesteps_survived if self.timesteps_survived else None,
            'mean_reward_1': self.sum_episode_rewards_2 / self.timesteps_survived if self.timesteps_survived else None,
            'fruits_eaten': (self.fruits_eaten_1, self.fruits_eaten_2),
            'termination_reason': self.termination_reason,
        }
        flat_stats.update({
            f'action_counter_{action}': self.action_counter.get(action, 0)
            for action in ALL_SNAKE_ACTIONS
        })
        return flat_stats

    def to_dataframe(self):
        """ Convert the episode statistics to a Pandas data frame. """
        return pd.DataFrame([self.flatten()])

    def __str__(self):
        return pprint.pformat(self.flatten())
