# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings('ignore')
# import keyboard

import matplotlib.pyplot as plt
import numpy as np
from env import Env
from tqdm import tqdm
import cv2
from collections import defaultdict

from ppo_agent import PPO
from ppo_agent import Memory
from constants import CONSTANTS
import torch

CONST = CONSTANTS()

np.set_printoptions(threshold=np.inf, linewidth=1000, precision=3, suppress=True)



env = Env()

memory = Memory()
rlAgent = PPO(env)
rlAgent.loadModel("checkpoints/ActorCritic.pt", 1)

NUM_EPISODES = 3
LEN_EPISODES = 1000
UPDATE_TIMESTEP = 2000
curState = []
newState = []
reward_history = []
mapNewVisPenalty_history = defaultdict(list)
totalViewed = []
dispFlag = False

# curRawState = env.reset()
# curState = rlAgent.formatInput(curRawState)
# rlAgent.summaryWriter_showNetwork(curState[0])

keyPress = 0
timestep = 0
loss = None


#whether per episode:
per_episode= 1

for episode in tqdm(range(NUM_EPISODES)):
    curRawState = env.reset()

    # generate state for each agent
    curState = rlAgent.formatInput(curRawState)

    if per_episode:
        env.out = cv2.VideoWriter(f"test_V/testVideo" + str(episode + 1) + ".avi", env.fourcc, 50, (700, 700))




    for step in range(LEN_EPISODES):


        env.render_test(step, None, episode, test= 1)
        env.save_test(step, None, episode, test= 1)


        # Get agent actions
        aActions = []
        for i in range(CONST.NUM_AGENTS):
            # get action for each agent using its current state from the network
            action = rlAgent.policy.act(curState[i], memory)
            #action = rlAgent.policy.act_test(curState[i])
            aActions.append(action)

        # do actions
        agentPosList, advrsyPosList, display, reward, newAreaVis, penalty, done = env.step(aActions)
        if step == LEN_EPISODES - 1:
            done = True


        # update nextState
        newRawState = []
        for agentPos in agentPosList:
            newRawState.append([agentPos, advrsyPosList, display])
        newState = rlAgent.formatInput(newRawState)





        # set current state for next step
        curState = newState

        if done:
            break

    # post episode


env.out.release()

