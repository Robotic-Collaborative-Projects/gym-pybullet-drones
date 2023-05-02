import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
# from gym_pybullet_drones.envs.DogAviary import DogAviary
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
# from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
# from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl   
# from gym_pybullet_drones.control.DogDroneControl import DogDroneControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool



DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 6
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 1000
DEFAULT_CONTROL_FREQ_HZ = 1000

DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False



class DroneEnvWrapper():
    def __init__(self, 
                 drone=DEFAULT_DRONES,
                 num_drones=DEFAULT_NUM_DRONES,
                 physics=DEFAULT_PHYSICS,
                 vision=DEFAULT_VISION,
                 gui=DEFAULT_GUI,
                 record_video=DEFAULT_RECORD_VISION,
                 plot=DEFAULT_PLOT,
                 user_debug_gui=DEFAULT_USER_DEBUG_GUI,
                 obstacles=DEFAULT_OBSTACLES,
                 simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
                 control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
                 duration_sec=None,
                 output_folder=DEFAULT_OUTPUT_FOLDER,
                 colab=DEFAULT_COLAB,
                 INIT_XYZS=None,
                 INIT_RPYS=None):
        
        AGGR_PHY_STEPS = int(simulation_freq_hz/DEFAULT_CONTROL_FREQ_HZ) if DEFAULT_AGGREGATE else 1
        #### Create the environment with or without video capture ##
        self.env = CtrlAviary(drone_model=drone,
                            num_drones=num_drones,
                            initial_xyzs=INIT_XYZS,
                            initial_rpys=INIT_RPYS,
                            physics=physics,
                            neighbourhood_radius=10,
                            freq=simulation_freq_hz,
                            aggregate_phy_steps=AGGR_PHY_STEPS,
                            gui=gui,
                            record=record_video,
                            obstacles=obstacles,
                            user_debug_gui=user_debug_gui,
                            output_folder='results'
                            )
        
        self.num_drones = num_drones
        self.init_position = INIT_XYZS
        self.init_orientation = INIT_RPYS

        self.ctrl = [DSLPIDControl(drone_model=DroneModel("cf2x")) for i in range(num_drones)]    

        self.position_cmd = INIT_XYZS
        self.rotation_cmd = INIT_RPYS
        self.action = {str(i): np.array([0,0,0,0]) for i in range(self.num_drones)}
        self.obs = self.env._computeObs()


    def update(self, move_drone=None, target_pos=None):    
        
        if move_drone != None:
            for j in range(self.num_drones):
                if j == move_drone:
                    self.action[str(j)], _, _ = self.ctrl[j].computeControlFromState(self.env.TIMESTEP,
                                                                state=self.obs[str(j)]["state"],
                                                                target_pos=target_pos,
                                                                target_rpy=self.init_orientation[j,:],
                                                                )
                else:
                    self.action[str(j)] = np.array([0,0,0,0])
        else: 
            self.action = {str(i): np.array([0,0,0,0]) for i in range(self.num_drones)}
        self.obs,_,_,_ = self.env.update(self.action)
            
        return self.obs
    
    def get_obs_no_update(self):
        return self.obs
    
    def close(self):
        self.env.close()

    def setCommand(self,rot, pos):
        self.position_cmd = pos
        self.rotation_cmd = rot

    def getImage(self, drone_idx, obs):
        rgb_img = obs[f'{drone_idx}']['rgb']
        seg_img = obs[f'{drone_idx}']['seg']
        depth_img = obs[f'{drone_idx}']['dep']

        return rgb_img, seg_img, depth_img
