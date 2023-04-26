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
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl   
# from gym_pybullet_drones.control.DogDroneControl import DogDroneControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool



DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 6
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 1000
DEFAULT_CONTROL_FREQ_HZ = 1000
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

num_drones = DEFAULT_NUM_DRONES
INIT_B1 = np.array([0,0,.5])
INIT_XYZS = np.zeros((num_drones,3))



for i in range(num_drones):
    INIT_XYZS[i,:] = np.array([[-.5+i,1,.0]])

print(INIT_XYZS)


INIT_RPYS = np.array([[0, 0,  0] for i in range(num_drones)])
AGGR_PHY_STEPS = int(DEFAULT_SIMULATION_FREQ_HZ/DEFAULT_CONTROL_FREQ_HZ) if DEFAULT_AGGREGATE else 1

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
                 duration_sec=DEFAULT_DURATION_SEC,
                 output_folder=DEFAULT_OUTPUT_FOLDER,
                 colab=DEFAULT_COLAB):
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
                            user_debug_gui=user_debug_gui
                            )
        
        self.num_drones = num_drones
        self.init_position = INIT_XYZS
        self.init_orientation = INIT_RPYS

        self.ctrl = [DSLPIDControl(drone_model=DroneModel("cf2x")) for i in range(num_drones)]    

        self.position_cmd = np.zeros((num_drones,3))
        self.rotation_cmd = np.zeros((num_drones,3))


    def update(self, Noise=False):    
        
        if Noise == False:
            action = {str(i): np.array([0,0,0,0]) for i in range(self.num_drones)}
            obs, reward, done, info = self.env.update(action)

            for j in range(self.num_drones):
                action[str(j)], _, _ = self.ctrl[j].computeControlFromState(self.env.TIMESTEP,
                                                                    state=obs[str(j)]["state"],
                                                                    target_pos=self.position_cmd[j,:],
                                                                    target_rpy=self.rotation_cmd[j,:],
                                                                    )
                
            return obs, reward, done, info
    
    def close(self):
        self.env.close()

    def setCommand(self,rot, pos, drone_idx):
        self.position_cmd[drone_idx,:] = pos
        self.rotation_cmd[drone_idx,:] = rot

    def getImage(self, drone_idx):
        drone_gt_state = self.obs[f'{drone_idx}']
        self.env._get_drone_images

        return rgb_img, seg_img, depth_img
