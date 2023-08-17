import gym
from cartpole_tutorial_DQN import Agent
from utils import plotLearning
import numpy as np


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, output_size=2, eps_end=0.01,
                  input_size=[4], lr=0.001)
    scores, eps_history = [], []
    high_score = 0
    n_games = 500
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)

        if i > 60 and score > high_score:
            agent.Q_eval.save("cart_tutorial_score.pth")

        if score > high_score:
            high_score = score
        
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    # filename = 'lunar_lander.png'
    # plotLearning(x, scores, eps_history, filename)
