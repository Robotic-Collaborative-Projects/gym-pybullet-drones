
import numpy as np
import pybullet as p
import time

# import B1 related files
from b1_wrapper.B1Wrapper import B1Wrapper
from b1_wrapper.b1_trot import trot
from utils import *
import cv2

from drones_wrapper import *

b1_params = {
    "plan_freq":     0.05,
    "update_time":   0.02,
    "sim_dt":       .001,
    "gait_params":   trot
}
SIM_DURATION = 120
SIM_FREQ = 1000

def run():    
    drones = DroneEnvWrapper()
    dog = B1Wrapper(b1_params)

    action = {str(i): np.array([0,0,0,0]) for i in range(drones.num_drones)}
    for i in range(0, int(SIM_DURATION*SIM_FREQ)):

        q,v = dog.update(np.array([1,0]))
        obs, reward, done, info = drones.update()
        p.stepSimulation()

    drones.close()


if __name__ == "__main__":
    run()