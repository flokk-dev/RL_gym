from stable_baselines3.common.callbacks import CheckpointCallback
import gym
import sys
import math
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

model_file = str(sys.argv[-4])
model_type = str(sys.argv[-3])
game_name = str(sys.argv[-2])
timesteps = int(sys.argv[-1])

env = make_atari_env(game_name, n_envs=16)
env = VecFrameStack(env, n_stack=4)

if model_type=="DQN":

    var_learn = int(float(timesteps)/10)
    model = DQN.load(model_file, env=env)
    model_name = 'dqn_second_'+game_name

    print(model.device)

if model_type=="A2C":

    model = A2C("MlpPolicy", env, verbose=1)
    model_name = 'a2c_'+game_name
    
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./ckpts/', name_prefix=model_name)
    
model.learn(total_timesteps=timesteps, progress_bar=True, callback=checkpoint_callback, reset_num_timesteps=False)
model.save(model_name)
