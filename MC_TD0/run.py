# Spring 2021, IOC 5269 Reinforcement Learning
# HW1-PartII: First-Visit Monte-Carlo and Temporal-difference policy evaluation

import gym
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


env = gym.make("Blackjack-v0")

def mc_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for a given policy using first-visit Monte-Carlo sampling
        
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
        
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
    
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate sample returns
            3. Iterate and update the value function
        ----------
        
    """
    
    # value function
    V = defaultdict(float)
    ##### FINISH TODOS HERE #####
    N=defaultdict(float)
    S=defaultdict(float)
    for count in range(num_episodes):
        observation = env.reset()
        episode=[]
        while True:
            action=policy(observation)
            next_observation, reward, done, _ = env.step(action)
            episode.append((observation,reward))
            if done:
                break
            observation=next_observation
        first={}
        gt=[]
        for i in range(len(episode)):
            gt.append(0)
        for i in range(len(episode)-1,-1,-1):
            s=episode[i]
            if i!=len(episode)-1:
                gt[i]=s[1]+gt[i+1]*gamma
            else:
                gt[i]=s[1]
        for i in range(0,len(episode)):
            s=episode[i]
            if s[0] not in first:
                first[s[0]]=1
                N[s[0]]+=1.0
                S[s[0]]+=gt[i]
                V[s[0]]=S[s[0]]/N[s[0]]
    #############################

    return V


def td0_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for the given policy using TD(0)
    
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
    
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate TD errors
            3. Iterate and update the value function
        ----------
    """
    # value function
    V = defaultdict(float)

    ##### FINISH TODOS HERE #####
    for count in range(num_episodes):
        observation = env.reset()
        episode=[]
        while True:
            action=policy(observation)
            next_observation, reward, done, _ = env.step(action)
            # V[observation]=V[observation]+1*(reward+gamma*V[next_observation]-V[observation]) 
            if done:
                V[observation]=V[observation]+0.1*(reward-V[observation])  
                break
            else:
                V[observation]=V[observation]+0.1*(reward+gamma*V[next_observation]-V[observation])   
            observation=next_observation

    

    #############################
    return V
    

    

def plot_value_function(V, title="Value Function"):
    """
        Plots the value function as a surface plot.
        (Credit: Denny Britz)
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
    
    
def apply_policy(observation):
    """
        A policy under which one will stick if the sum of cards is >= 20 and hit otherwise.
    """
    #0 stick 1 hit
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


if __name__ == '__main__':

    
    V_mc_10k = mc_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_mc_10k, title="MC 10,000 Steps")
    V_mc_500k = mc_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_mc_500k, title="MC 500,000 Steps")

    V_td0_10k = td0_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_td0_10k, title="TD(0) 10,000 Steps")
    V_td0_500k = td0_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_td0_500k, title="TD(0) 500,000 Steps")

    



