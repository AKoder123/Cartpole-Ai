
import gym  # pip install gym==0.25.2
from existing_tutorial import Agent
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
# agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2, eps_end=0.01,
#                   input_dims=[4], lr=0.001)

agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, output_size=2, eps_end=0.01,
                input_size=[4], lr=0.001)
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    while not done:
        env.render()
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
