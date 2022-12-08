import gym

for register in gym.envs.registry.all():
    print(register)
