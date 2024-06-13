import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from nn import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer
from agent import Agent

# set up a td3 agent to train in a simulated robotics environment
if __name__ == '__main__':

    # save the models
    if not os.path.exists('models/td3'):
        os.makedirs('models/td3')

    # set and configure the robot env
    env_name = 'Door'
    env = suite.make(
        env_name,
        robots=['Panda'],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        use_camera_obs=False, # to put cnn to train over camera imgs
        horizon=300,
        reward_shaping=True, # if false give robot reqard only when successfully opens the door
        control_freq=20,
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    # create the TD3 agent with the specified parameters
    agent = Agent(actor_learning_rate=actor_learning_rate,
                  critic_learning_rate=critic_learning_rate,
                  tau=0.005,
                  input_dims=env.observation_space.shape,
                  env=env,
                  n_actions=env.action_space.shape[0],
                  layer1_size=layer1_size,
                  layer2_size=layer2_size,
                  batch_size=batch_size)
    
    # initialize tensor board writer for logging
    writter = SummaryWriter('logs')
    n_games = 1000 
    best_score = 0
    episode_identifier = f'0: actor_learning_rate={actor_learning_rate}, critic_learning_rate={critic_learning_rate}, layer1_size={layer1_size}, layer2_size={layer2_size}'

    agent.load_models()

    # start training loop
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

        # push score to tensorboard logs
        writter.add_scalar(f'score - {episode_identifier}', score, global_step=i)

        # save the model every 10 epochs
        if (i % 10):
            agent.save_models()

        print(f'episode: {i}, score: {score}')


