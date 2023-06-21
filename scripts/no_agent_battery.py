import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import wandb

import sinergym
from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *

# Environment ID
environment = "Eplus-office-mixed-discrete-v1"
#Name of the experiment
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
experiment_name = 'no-agent-' + environment
experiment_name += '_' + experiment_date

# Create wandb.config object in order to log all experiment params
experiment_params = {
    'sinergym-version': sinergym.__version__,
    'python-version': sys.version
}
experiment_params.update({'environment':environment,
                          'episodes':1,
                          'algorithm':'no-agent'})

# Get wandb init params (you have to specify your own project and entity)
wandb_params = {"project": 'zephframe-team',
                "entity": 'zephframeteam'}

# # Init wandb entry
# run = wandb.init(
#     name=experiment_name + '_' + wandb.util.generate_id(),
#     config=experiment_params,
#     ** wandb_params
# )

env = gym.make(
    environment,
    action_variables=[],
    action_space=gym.spaces.Box(
        low=0,
        high=0,
        shape=(0,)),
    action_definition=None)
env = LoggerWrapper(env)

for i in range(1):
    obs, info = env.reset()
    rewards = []
    terminated = False
    current_month = 0
    while not terminated:
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))
env.close()

# artifact = wandb.Artifact(
#         name="training",
#         type="experiment1")
# artifact.add_dir(
#         env.simulator._env_working_dir_parent,
#         name='training_output/')
# run.log_artifact(artifact)

# # wandb has finished
# run.finish()