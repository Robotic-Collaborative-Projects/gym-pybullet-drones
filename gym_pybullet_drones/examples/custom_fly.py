
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
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
SIM_DURATION = 15
SIM_FREQ = 1000

num_drones = 6
INIT_B1_pos = np.array([.5,0,.5])
B1_vel = np.array([.3,0,0])
B1_w = 0

INIT_drones = np.zeros((num_drones,3))
ctr = 0
for i in range(num_drones):

    if i % 2 == 0:
        INIT_drones[i,:] = np.array([[ctr,1,.0]])
    else: 
        INIT_drones[i,:] = np.array([[ctr,-1,.0]])
        ctr +=1

INIT_drones_rots = np.array([[0, 0,  0] for i in range(num_drones)])

GT_states = np.zeros((int(SIM_DURATION*SIM_FREQ), 6))
est_states = np.zeros_like(GT_states)
cov = np.zeros_like(GT_states)

def run():    
    drones = DroneEnvWrapper(INIT_XYZS=INIT_drones, INIT_RPYS=INIT_drones_rots, simulation_freq_hz=SIM_FREQ, duration_sec=SIM_DURATION)
    dog = B1Wrapper(b1_params, initial_position=INIT_B1_pos)
    dog.set_cmd(B1_vel,B1_w)
    sensorModel = EKFModel(INIT_B1_pos, B1_vel, INIT_drones, .001)

    for i in range(0, int(SIM_DURATION*SIM_FREQ)):
        start = time.time()

        if i % 1000 == 0:
            q, v , rgb, depth, seg = dog.update(get_image_bool=True) #
        else:
            q,v = dog.update(get_image_bool=False)
        
        
        
        obs = drones.update()
        sens_state, sens_cov = sensorModel.compute_EKF(q[0:3], obs)
        GT_states[i,:] = np.append(q[:3], v[:3])
        est_states[i,:] = sens_state
        cov[i,:] = sens_cov.diagonal()

    print('yay')    
    np.savez('results_out', GT_states, est_states, cov)
    
    drones.close()
    


if __name__ == "__main__":
    run()