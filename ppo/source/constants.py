	# -*- coding: utf-8 -*-
# author Amrish Bakaran
# Copyright
# Constants

import numpy as np
RENDER_PYGAME = True



## Drone Constants
class CONSTANTS:
    def __init__(self):
        self.TIME_STEP=1
        
        
        self.NUM_AGENTS = 1
        self.MAX_AGENT_VEL= 1   
        
        self.NUM_ADVRSRY = 1
        
        self.GRID_SZ = self.MAX_AGENT_VEL * self.TIME_STEP * 1.0

        self.MAX_STEPS = 50
        
        self.MAP_SIZE = self.MAX_STEPS * self.GRID_SZ
        
        self.GRID_CENTER_PTS = self.getGridCenterPts()
        
        self.SEPERATION_PENALTY = 10
        
    def getGridCenterPts(self):
        x, y = np.meshgrid(np.arange(50), np.arange(50))
        x, y = x.flatten()+0.5, y.flatten() + 0.5
        points = np.vstack((x,y)).T
        return points

## Area
        


