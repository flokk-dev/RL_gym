import gym
import sys
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import cv2
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

stack = str(sys.argv[-6])=='True'
evaluate = str(sys.argv[-5])=='True'
model_type = str(sys.argv[-4])
model_name = str(sys.argv[-3])
game_name = str(sys.argv[-2])
record = str(sys.argv[-1])=='True'

env = make_atari_env(game_name, n_envs=16)
env = VecFrameStack(env, n_stack=4)

if stack:
    env = make_atari_env(game_name, n_envs=16)
    env = VecFrameStack(env, n_stack=4)
else:
    env = make_atari_env(game_name, n_envs=1)

if model_type=="DQN":
  model = DQN.load(model_name)
elif model_type=="A2C":
  model = A2C.load(model_name)

if record:
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (160, 210), True)

if evaluate:
  obs = env.reset()
  print('MODEL EVALUATION')
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)
  print(f'mean REWARD: {mean_reward}  | std REWARD: {std_reward}')



obs = env.reset()

iter = 0
while True:
    #random
    #action = env.action_space.sample()

    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    img = env.render(mode="rgb_array")
    img_alone = img[:210,:160,:]
    cv2.imshow('game', img)
    cv2.waitKey(130)

    if record:
      if iter<=12000:
        out.write(img_alone)
      elif iter == 1200:
        print('RECORD ENDED')
        out.release()

    if iter%100==0:
      print(f'iter: {iter}')
    iter += 1
    if done.all():
      obs = env.reset()
    
cv2.destroyAllWindows()
