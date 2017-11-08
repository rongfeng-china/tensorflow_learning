from maze_env import Maze
from RL_brain import DeepQNetwork
def run_maze():
    step = 0 
    for episode in range(300):
        # initialization
        observation = env.reset()

        while True:
            env.render()
            ## choose an action            
            action = RL.choose_action(observation)
            ## feedback from the environment
            observation_, reward,done = env.step(action)
            ## at first, store in memory ( for 200 steps), after that start to learn
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step %5 == 0):
                RL.learn()

            observation = observation_

            if done:
                break
            step += 1

    print ('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                     learning_rate = .01,
                     reward_decay = .9,
                     e_greedy = .9,
                     replace_target_iter = 200,
                     memory_size = 2000,
                     )
    env.after(100,run_maze)
    env.mainloop()
    RL.plot_cost()


