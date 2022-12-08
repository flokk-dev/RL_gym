import gym
import sys
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import json

model_type = str(sys.argv[-2])
game_name = str(sys.argv[-1])

this_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'./ckpts/')
list_files_solo = os.listdir(this_path)

if model_type=="DQN":
  list_files = list(filter(lambda x: x.startswith('dqn'),list_files_solo))
elif model_type=="A2C":
  list_files = list(filter(lambda x: x.startswith('a2c'),list_files_solo))

list_files = list(map(lambda x: 'ckpts'+os.sep+str(x), list_files))
tests = {}

env = make_atari_env(game_name, n_envs=16)
env = VecFrameStack(env, n_stack=4)

for model_file in list_files:

  if model_type=="DQN":
    model = DQN.load(model_file)
  elif model_type=="A2C":
    model = A2C.load(model_file)


  obs = env.reset()
  print(f'EVALUATING {model_file}')
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
  print(f'mean REWARD: {mean_reward}  | std REWARD: {std_reward}')
  obs = env.reset()
  tests[model_file] = (mean_reward, std_reward)

out_file = open('tests.json', 'w')
json.dump(tests, out_file)
out_file.close()
