# -*- coding: utf-8 -*-

import numpy as np
from env import Env
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import random as rand

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)

def getKeyPress():
    wait = True
    while(wait):
        k = cv2.waitKeyEx(1)
        """
        if k != -1:
            print(k)
        """
        if k == 119:
            act = 1
            wait = False
        elif k == 97:
            act = 2
            wait = False
        elif k == 115:
            act = 3
            wait = False
        elif k == 100:
            act = 4
            wait = False
    return act


env = Env()

NUM_EPISODES = 100
LEN_EPISODES = 200

act = 0

env.render()



for episode in tqdm(range(NUM_EPISODES)):
    env.reset()
    env.render_test(0, None, episode)







    for step in range(1000000):
        # step agent
        actions = [getKeyPress()]
        agentPosList, advrsyPosList, display, reward, newAreaVis, penalty, done = env.step(actions)
#        print(rewardList, done)
#        print(agentPosList)


        print(np.min(display))
        print(reward)

        #print(display)








        env.render_test(step,None, episode)
        cv2.waitKey(1)

        if done:
            # end episode
            print("Episode Complete")
            break


