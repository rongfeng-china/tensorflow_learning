import gym
from RL_brain import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

#print(env.action_space)
print(env.observation_space.shape)
#print(env.observation_space.high)
#print(env.observation_space.low)


RL = DeepQNetwork(n_actions = env.action_space.n, 
            n_features = env.observation_space.shape[0],
            learning_rate = 0.01, e_greedy = 0.9,
            replace_target_iter = 100, memory_size = 2000,
            e_greedy_increment = .0008)

total_steps = 0

for i_episode in xrange(100):
    ## initialization
    observation = env.reset()
    ep_r = 0

    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        ## feedback from the environment
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold-abs(x))/env.x_threshold - .8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - .5
        r = r1 + r2
        
        ## store in memory    
        RL.store_transition(observation,action,r,s_)
 
        if total_steps < 200 and (total_steps % 5 == 0):
            RL.learn()
