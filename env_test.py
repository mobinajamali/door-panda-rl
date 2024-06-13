import time
import os
import gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from agent import Agent

# test how the agent is doing
if __name__ == '__main__':

    if not os.path.exists('tmp/td3'):
        os.makedirs('tmp/td3')

    # env setup
    env_name = 'Door'
    env = suite.make(
        env_name,
        robots=['Panda'],
        controller_configs=suite.load_controller_config(default_controller='JOINT_VELOCITY'),
        has_renderer=True,
        use_camera_obs=False,
        horizon=75,
        render_camera='frontview', # show some rendering on screen
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)


    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    # agent setup
    agent = Agent(actor_learning_rate=actor_learning_rate,
                  critic_learning_rate=critic_learning_rate,
                  tau=0.005,
                  input_dims=env.observation_space.shape,
                  env=env,
                  n_actions=env.action_space.shape[0],
                  layer1_size=layer1_size,
                  layer2_size=layer2_size,
                  batch_size=batch_size)
    

    n_games = 3
    best_score = 0
    episode_identifier = f'0: actor_learning_rate={actor_learning_rate}, critic_learning_rate={critic_learning_rate}, layer1_size={layer1_size}, layer2_size={layer2_size}'

    agent.load_models()

    # training loop
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, validation=True)
            next_observation, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = next_observation
            #time.sleep(0.03)

        print(f'episode: {i}, score: {score}')