import gym
from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped


RL = DeepQNetwork(n_actions = env.action_space.n,
        n_features = env.observation_space.shape[0],
        learning_rate = .001, e_greedy = .9,
        replace_target_iter = 300, memory_size = 3000,
        e_greedy_increment = .0002)

total_steps = 0

for i_episode in range(10):
    ## initialization
    observation = env.reset()
    ep_r = 0

    while True:
        ## update the screen
        env.render() 
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        ## feedback from environment
        position, velocity = observation_
        r = abs(position - (-.5))  # r in [0,1]

        ## store in memory
        RL.store_transition(observation,action,r,observation_)

        # reward for episode
        ep_r += r

        if total_steps > 1000:
            RL.learn()

        if done:
            print ('i_episode: '+str(i_episode))
            print ('ep_r: %f' %(round(ep_r,2)))
            print ('epsilon: %f' %(round(RL.epsilon,2)))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
