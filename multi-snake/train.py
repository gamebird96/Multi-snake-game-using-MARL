#!/usr/bin/env python3

""" Front-end script for training a Snake agent. """

import json
import sys

from keras.models import Sequential, Input, Model
from keras.layers import *
from keras.optimizers import *

from snakeai.gameplay.environment import Environment
from snakeai.utils.cli import HelpOnFailArgumentParser
from snakeai.agent.minimaxdqn import MinimaxDeepQNetworkAgent
from snakeai.agent.minimaxsingledqn import MinimaxSingleDeepQNetworkAgent


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        required=True,
        type=int,
        default=30000,
        help='The number of episodes to run consecutively.',
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


def create_dqn_model(env, num_last_frames):


    inputs = Input(shape=(num_last_frames, ) + env.observation_shape)
    hidden = Conv2D(16, kernel_size=(3,3), strides=(1,1), data_format='channels_first', activation='relu')(inputs)
    hidden = Conv2D(32, kernel_size=(3,3), strides=(1,1), data_format='channels_first', activation='relu')(hidden)
    hidden = Flatten()(hidden)
    hidden = Dense(256, activation='relu')(hidden)
    outputs = Dense(env.num_actions*env.num_actions, activation='linear')(hidden)
    model = Model(input=[inputs], output=[outputs])

    model.summary()
    model.compile(RMSprop(), 'MSE')

    return model


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    env = create_snake_environment(parsed_args.level)
    model_snake_1 = create_dqn_model(env, num_last_frames=4)
    model_snake_2 = create_dqn_model(env, num_last_frames=4)
    
    agent = MinimaxDeepQNetworkAgent(
        model_1 = model_snake_1,
        model_2 = model_snake_2,
        memory_size = -1,
        num_last_frames = model_snake_1.input_shape[1]
    )
    agent.train(
        env,
        batch_size=64,
        num_episodes=parsed_args.num_episodes,
        checkpoint_freq=parsed_args.num_episodes // 10,
        discount_factor=0.98
    )
    #model = create_dqn_model(env, num_last_frames=4)

    '''
    agent = MinimaxSingleDeepQNetworkAgent(
        model = model_snake_1,

        memory_size = -1,
        num_last_frames = model_snake_1.input_shape[1]
    )
    agent.train(
        env,
        batch_size=64,
        num_episodes=parsed_args.num_episodes,
        checkpoint_freq=parsed_args.num_episodes // 10,
        discount_factor=0.98
    )
    '''

if __name__ == '__main__':
    main()
