import gym
from RL_brain import PolicyGradient

DISPLAY_REWARD_THRESHOLD = 400

env = gym.make('CartPole-v0')
env = env.unwrapped

#print(env.action_space)
#print(env.observation_space.shape)
#print(env.observation_space.high)
#print(env.observation_space.low)

RL = PolicyGradient(n_actions = env.action_space.n, 
            n_features = env.observation_space.shape[0],
            learning_rate = 0.02, 
            reward_decay - .99)

total_steps = 0

for i_episode in xrange(1000):
    ## initialization
    observation = env.reset()

    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        ## store in memory    
        RL.store_transition(observation,action,r,observation_)
 
        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * .99 + ep_rs_sum * .01
        
            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break
        
        observation = observation_
        total_step += 1

