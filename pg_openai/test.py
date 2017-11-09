import gym
from RL_brain import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

#print(env.action_space)
print(env.observation_space.shape)
print(env.x_threshold)
#print(env.observation_space.high)
#print(env.observation_space.low)



