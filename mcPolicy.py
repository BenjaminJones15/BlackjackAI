# -*- coding: utf-8 -*-
"""
Updated Thu Dec 15 11:43:20 2022

@author: Cameron Leach
cleach4
"""


"""
This runs an Monte Carlo policy evaluation on a blackjack game modeled by GYM
Some dependecies are neccesary and this code is repurposed from these sources for our testing
https://towardsdatascience.com/learning-to-win-blackjack-with-monte-carlo-methods-61c90a52d53e
"""
import random

import sys
import gym
import numpy
from collections import defaultdict
import matplotlib.pyplot


"""
The policy for the simulation. Currently just weights combined value of hands and what the dealer is showing.
"""
def policy(total):
    moves = [0,1]
    if total >= 17:
        move = random.choices(moves, weights=(85,15), k=1)
        return move[0]
    else:
        move = random.choices(moves, weights=(25,75), k=1)
        return move[0]


"""
Creates the blackjack game for simulation using the gym model
https://www.gymlibrary.dev/
"""
blackjack = gym.make('Blackjack-v1')




def playGame(blackjack):
    """
    Plays a hand with the defined policy
    """
    game = []
    state = blackjack.reset()
    play = True
    while play:
        action = policy(state[0])
        state1, reward, done, empty= blackjack.step(action)
        game.append((state1, action, reward))
        state = state1
        if done:
            play = False
    return game

def Qvalue(game, Qval, rewardSum, seen):
    gamma = 1

    for s, a, r in game:
            firstOccurence = next(i for i,x in enumerate(game) if x[0] == s)
            G = sum([x[2]*(gamma**i) for i,x in enumerate(game[firstOccurence:])])
            rewardSum[s][a] += G
            seen[s][a] += 1.0
            Qval[s][a] = rewardSum[s][a] / seen[s][a]



def mc_predict(blackjack, numGames):

    """
    This is the primary method. Plays through several episodes of the environment. 
    """
    rewardSum = defaultdict(lambda: numpy.zeros(blackjack.action_space.n))
    seen = defaultdict(lambda: numpy.zeros(blackjack.action_space.n))
    Qval = defaultdict(lambda: numpy.zeros(blackjack.action_space.n))
    evaluations = []
    
    for gme in range(1, numGames+1):
        if gme % 500 == 0:
            evaluations.append(evalP(Qval))
            print("\rGames {}/{}.".format(gme, numGames), end="")
            sys.stdout.flush()
            
        game = playGame(blackjack)

        Qvalue(game, Qval, rewardSum, seen)
            
    return Qval, evaluations

def evalP(Qval, games=10000):
    """
    Evaluate the policy of our agent with this
    """
    wins = 0
    for x in range(games):
        state = blackjack.reset()
        
        done = False
        while not done:
            action = numpy.argmax(Qval[state])
            
            state, reward, done, empty = blackjack.step(action=action)
            
        if reward > 0:
            wins += 1
        
    return wins / games



#predict the policy values for our test policy
Qvals,evaluations = mc_predict(blackjack, 300000)



matplotlib.pyplot.plot([i * 1000 for i in range(len(evaluations))], evaluations)
matplotlib.pyplot.xlabel('game')
matplotlib.pyplot.ylabel('win percentage')

print()
print(numpy.average(evaluations))







