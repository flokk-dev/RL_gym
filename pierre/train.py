from stable_baselines3.common.callbacks import CheckpointCallback
import gym
import sys
import math
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

stack = str(sys.argv[-4]) == 'True'
model_type = str(sys.argv[-3])
game_name = str(sys.argv[-2])
timesteps = int(sys.argv[-1])

if stack:
    env = make_atari_env(game_name, n_envs=16)
    env = VecFrameStack(env, n_stack=4)
else:
    env = make_atari_env(game_name, n_envs=1)

if model_type == "DQN":
    var_learn = int(float(timesteps) / 10)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-4,
        batch_size=64,
        buffer_size=100000,
        learning_starts=1000000,
        gamma=0.97,
        target_update_interval=100,
        train_freq=16,
        gradient_steps=4,
        exploration_fraction=0.16,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[256, 256]),
        device='cuda'
    )

    model_name = 'dqn_256' + game_name

if model_type == "A2C":
    model = A2C("MlpPolicy", env, verbose=1)
    model_name = 'a2c_' + game_name

checkpoint_callback = CheckpointCallback(save_freq=500000, save_path='./ckpts/',
                                         name_prefix=model_name)

model.learn(total_timesteps=timesteps, progress_bar=True, callback=checkpoint_callback,
            reset_num_timesteps=False)
model.save(model_name)
