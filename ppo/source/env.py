# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# brief Environment class for the simulation

import numpy as np
import random
import copy
import math
import cv2

from constants import CONSTANTS as K
CONST = K()
from agent import Agent
from obstacle import Obstacle
obsMap = Obstacle()

from agent import Agent as Adversary

np.set_printoptions(precision=3, suppress=True)
class Env:
    def __init__(self):
        self.isNewSess = True
        self.timeStep = CONST.TIME_STEP
        self.obsMaps, self.vsbs, self.vsbPolys = self.initObsMaps_Vsbs()
        self.obstacleMap , self.vsb, self.vsbPoly, self.mapId = self.setRandMap_vsb()
        # modified: decay rate:
        self.decay= 2
        # modified: cap the upperbound of penalty
        self.cap= 200




        self.obstacleMap , self.vsb, self.vsbPoly, self.mapId = self.setRandMap_vsb()
        self.obstacleMap,self.obsPlusViewed, self.currentMapState, self.agents, self.adversaries = self.initTotalArea_agents(CONST.NUM_AGENTS)

        # modified: reward map
        #self.rewardMap= np.where(self.obstacleMap == 0, 0, self.obstacleMap)


        #modified: previous sum of reward
        #self.prevSumR = np.sum(self.rewardMap)



        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(f"checkpoints/cnn1.avi",self.fourcc, 50, (700,700))
        self.out_test= cv2.VideoWriter(f"./testVideo.avi",self.fourcc, 50, (700,700))

    
    def initObsMaps_Vsbs(self):
        return obsMap.getAllObs_vsbs(np.zeros((50,50)))
    
    def setRandMap_vsb(self):
        i = random.randint(0, len(self.obsMaps)-1)
        return self.obsMaps[i], self.vsbs[i], self.vsbPolys[i], i
    
    def initTotalArea_agents(self, numAgents):
        # unviewed = 0
        # viewed = 255
        # obstacle = 150
        # agent Pos = 100
        # adversary Pos = 200
        
        obstacleMap = self.obstacleMap
        obstacleViewedMap = np.copy(obstacleMap)
        
        #initialize agents at random location
        agents = []
        x,y = np.nonzero(obstacleMap == 0)
        ndxs = random.sample(range(x.shape[0]), CONST.NUM_AGENTS + CONST.NUM_ADVRSRY)

        for ndx in ndxs[:-(CONST.NUM_ADVRSRY)]:
            agents.append(Agent(x[ndx]+0.5, y[ndx]+0.5))
#            agents.append(Agent())

        adversaries = []
        for ndx in ndxs[-(CONST.NUM_ADVRSRY):]:
            adversaries.append(Adversary(x[ndx]+0.5, y[ndx]+0.5))
#            adversaries.append(Adversary(25.5,10.5))
                   
        for agent in agents:
            obstacleViewedMap = self.vsb.updateVsbOnImg([agent.getState()[0]],obstacleViewedMap, self.vsbPoly)
        
        agentPos = [agent.getState()[0] for agent in agents]
        gPos = self.cartesian2Grid(agentPos)
        temp = self.updatePosMap(gPos, obstacleViewedMap, 100)

        currentMapState= temp

        """ Adversary positoin 
        advrsyPos = [adversary.getState()[0] for adversary in adversaries]
        advPos = self.cartesian2Grid(advrsyPos)
        currentMapState = self.updatePosMap(advPos, temp, 200)
        """
        
        return obstacleMap, obstacleViewedMap, currentMapState, agents, adversaries
    
    def resetTotalArea(self):
        obstacleMap = self.obstacleMap
        obstacleViewedMap = np.copy(obstacleMap)
        for agent in self.agents:
            obstacleViewedMap = self.vsb.updateVsbOnImg([agent.getState()[0]],obstacleViewedMap, self.vsbPoly)
        
        agentPos = [agent.getState()[0] for agent in self.agents]
        gPos = self.cartesian2Grid(agentPos)
        temp = self.updatePosMap(gPos, obstacleViewedMap, 100)

        currentMapState= temp


        #adversary position
        """ 
        advrsyPos = [adversary.getState()[0] for adversary in self.adversaries]
        advPos = self.cartesian2Grid(advrsyPos)
        currentMapState = self.updatePosMap(advPos, temp, 200)
        
        """
        
        return obstacleMap, obstacleViewedMap, currentMapState
    
    def initAgents(self, n):
        agents = []
        for i in range(0,n):
            agents.append(Agent())
        return agents
      
    def reset(self):
        
        # need to update initial state for reset function
        self.obstacleMap , self.vsb, self.vsbPoly, self.mapId = self.setRandMap_vsb()
        self.obstacleMap,self.obsPlusViewed, self.currentMapState, self.agents, self.adversaries = self.initTotalArea_agents(CONST.NUM_AGENTS)

        # modified: reward map
        #self.rewardMap = np.where(self.obstacleMap == 0, 0, self.obstacleMap)

        # modified: previous sum of reward
        #self.prevSumR = np.sum(self.rewardMap)
        
        state = []
        for agent in self.agents:
            state.append([agent.getState()[0],self.currentMapState])
        
        return state
        
    def getActionSpace(self):
        return [0,1,2,3,4]
    
    def getStateSpace(self):
        return self.obstacleMap.size
    
    def stepAgent(self, actions):
        # have to decide on the action space
        # waypoints or velocity
        posOut = []
        velOut = []
        for agent, action in zip(self.agents, actions):
            curState = agent.getState()
            futureState = copy.deepcopy(curState[0])
            if action == 0:
                    pass
            elif action == 1:
                futureState[1] += 1
            elif action == 2:
                futureState[0] += -1
            elif action == 3:
                futureState[1] += -1
            elif action == 4:
                futureState[0] += 1
            # check if agent in obstacle
            isValidPt = False
            if 0<futureState[0] <50 and 0<futureState[1] <50 :
                if self.obstacleMap[int(futureState[0]), int(futureState[1])] == 150:
                    isValidPt = True
            if 0<futureState[0] <50 and 0<futureState[1] <50 and not isValidPt:
                vel = np.array([0,0])
                if action == 0:
                    pass
                elif action == 1:
                    vel[1] = 1
                elif action == 2:
                    vel[0] = -1
                elif action == 3:
                    vel[1] = -1
                elif action == 4:
                    vel[0] = 1
                agent.setParams(vel)
                agent.updateState(self.timeStep)
                curState = agent.getState()
                posOut.append(curState[0])
                velOut.append(curState[1])
            else:
                posOut.append(curState[0])
                velOut.append(curState[1])
        return posOut, velOut
    
    def step(self, agentActions):
        agentPos, agentVel = self.stepAgent(agentActions)
        gPos = self.cartesian2Grid(agentPos)
        # get new visibility and update obsPlusViewed
        self.obsPlusViewed = self.vsb.updateVsbOnImg(agentPos,self.obsPlusViewed, self.vsbPoly)

        #Genearete currentStateMap with decay reward:

        # 150 is the wall, 255 (->0) (is newly viewed), initial unviewed is 0,  all pixels except wall is < 0, agent 100

        #update the new current viewed area from updateVsbPloygon set current viewed to 255
        self.currentMapState= self.vsb.updateVsbPolyOnImg(agentPos,self.currentMapState)

        # If pixel <= 0 (not wall (150), agent (100), current viewed (255->0)) decrease by decay rate
        self.currentMapState = np.where((self.currentMapState <= 0), self.currentMapState - self.decay, self.currentMapState)


        # set current under viewed pixels(255) to 0, after decay complete
        self.currentMapState= np.where((self.currentMapState == 255), 0, self.currentMapState)

        # apply the lowerbound cap for the penalty
        self.currentMapState = np.where((self.currentMapState < (-1 * self.cap)), -1 * self.cap, self.currentMapState)


        # update position on currentMapState
        self.currentMapState= self.updatePosMap(gPos, self.currentMapState, 100)


        #Adversary, left unused for now
        """
        advrsyPos = [adversary.getState()[0] for adversary in self.adversaries]
        advPos = self.cartesian2Grid(advrsyPos)
        self.currentMapState = self.updatePosMap(advPos, temp, 200)

        
        AdvVisibility = self.vsb.checkPtInVsbPoly(advrsyPos, agentPos)
        """

        #currently set adversary visibility to false : does't consider
        AdvVisibility= False
        
        display = self.currentMapState
        # update reward mechanism
        sumR, penalty = self.getReward(AdvVisibility)
        reward = sumR + penalty
        #done = np.count_nonzero(self.currentMapState==0) == 0
        done= False
        return agentPos, display, reward, sumR, penalty, done

    def render(self, show = 0):
        cap= self.cap

        img = np.copy(self.currentMapState)

        reward_map = img



        """ initialize heatmap """
        heatmap = cv2.resize(reward_map, (700, 700), interpolation=cv2.INTER_AREA)
        heatmapshow = np.rot90(heatmap, 1)

        heatmapshow = np.where(heatmapshow == 150 , 20, heatmapshow)
        heatmapshow = np.where(heatmapshow < 0, -1 * heatmapshow * 255 / cap, -1 * heatmapshow)
        heatmapshow = np.where(heatmapshow >= 200, 255, heatmapshow)




        heatmapshow = heatmapshow.astype(np.uint8)



        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

        if (show == 1):
            cv2.imshow("Heatmap", heatmapshow)
            cv2.waitKey(1)

        return  heatmapshow



    def parse_action(self, action):
        if action == 0:
            return 'stay'
        elif action == 1:
            return 'up'
        elif action == 2:
            return 'left'
        elif action == 3:
            return 'down'
        else:
            return 'right'


    def render_test(self, step, aActions, episode):
        self.generate_heat_text(aActions, step, episode, 1)

    def save_test(self, step, aActions, episode):


        displayImg = self.generate_heat_text(aActions, step, episode)

        self.out_test.write(displayImg.astype('uint8'))





    def generate_heat_text(self,  action, count, episode, show= 0):
        """ initialize heatmap """
        heatmapshow = self.render()

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org2 = (450, 15)
        org1 = (10, 15)
        org4 = (10, 45)
        org3 = (450, 45)

        # fontScale
        fontScale = 0.6

        # Blue color in BGR
        color = (0, 255, 0)

        # Line thickness of 2 px
        thickness = 2

        reward, penalty = self.getReward(False)

        # Using cv2.putText() method

        heatmapshow = cv2.putText(heatmapshow, "current episode: " + str(episode), org1, font,
                                  fontScale, color, thickness, cv2.LINE_AA)

        heatmapshow = cv2.putText(heatmapshow, "current step: " + str(count), org4, font,
                                  fontScale, color, thickness, cv2.LINE_AA)

        heatmapshow = cv2.putText(heatmapshow, "total penalty: " + str(reward + penalty), org2, font,
                                  fontScale, color, thickness, cv2.LINE_AA)

        word = self.parse_action(action)
        heatmapshow = cv2.putText(heatmapshow, "action: " + word, org3, font,
                                  fontScale, color, thickness, cv2.LINE_AA)

        if(show == 1):
            cv2.imshow("Heatmap for testLoop", heatmapshow)
            cv2.waitKey(1)

        return heatmapshow

    def save2Vid(self):

        
        displayImg = self.render()
        
        self.out.write(displayImg.astype('uint8'))
#        cv2.imshow("raw", displayImg)
#        cv2.waitKey(1)
            
    def updatePosMap(self, gPos, obsPlusViewed, val):
        currMapState = np.copy(obsPlusViewed)
        for pos in gPos:
            currMapState[pos[0],pos[1]] = val
        return currMapState


    def getReward(self, AdvVisibility):
        #150 is the wall, 255 (->0) (is newly viewed), initial unviewed is 0,  all pixels except wall is < 0, agent 100

        #sum up reward on all free pixels
        actualR = np.where((self.currentMapState<= 0), self.currentMapState, 0)
        curSumR = np.sum(actualR)


        penalty = 0
        if AdvVisibility:
            penalty += 0
        #            print("Visible")
        else:
            #            print("Not Visible")
            pass

        return curSumR, penalty


        





        
    def cartesian2Grid(self, posList):
        gridList = []
        for pos in posList:
            _x = math.floor(pos[0]/CONST.GRID_SZ)
            _y = math.floor(pos[1]/CONST.GRID_SZ)
            gridList.append([int(_x),int(_y)])
        return gridList
        
                    


