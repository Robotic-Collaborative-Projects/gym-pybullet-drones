
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import time

# import B1 related files
from b1_wrapper.B1Wrapper import B1Wrapper
from b1_wrapper.b1_trot import trot
from utils import *
import cv2
from klampt.model import trajectory

from drones_wrapper import *

b1_params = {
    "plan_freq":     0.05,
    "update_time":   0.02,
    "sim_dt":       .001,
    "gait_params":   trot
}
SIM_DURATION = 30
SIM_FREQ = 1000



num_drones = 6
INIT_B1_pos = np.array([.1,0,.5])
B1_vel = np.array([.3,0,0])
B1_w = 0

INIT_drones = np.zeros((num_drones,3))
ctr = 0
for i in range(num_drones):

    if i % 2 == 0:
        INIT_drones[i,:] = np.array([[.5*ctr,1,.0]])
    else: 
        INIT_drones[i,:] = np.array([[.5*ctr,-1,.0]])
        ctr +=1

traj = trajectory.Trajectory(milestones=[[0,0,0], [ctr/4, 0, .4], [ctr/2, 0, 0]])
herm = trajectory.HermiteTrajectory()
herm.makeSpline(traj)

INIT_drones_rots = np.array([[0, 0,  0] for i in range(num_drones)])

GT_states = []
est_states = []
cov = []

def run():    
    drones = DroneEnvWrapper(INIT_XYZS=INIT_drones, INIT_RPYS=INIT_drones_rots, simulation_freq_hz=SIM_FREQ, duration_sec=SIM_DURATION)
    dog = B1Wrapper(b1_params, initial_position=INIT_B1_pos)
    dog.set_cmd(B1_vel,B1_w)
    sensorModel = EKFModel(INIT_B1_pos, B1_vel, INIT_drones, .001)
    obs = drones.get_obs_no_update()
    q, v = dog.get_b1_state()
    hold_and_fly = False
    counter = 0
    flight_time = 0
    next_drone = 0
    delay = 0

    for i in range(0, int(SIM_DURATION*SIM_FREQ)):
            
        if i == 0: #compute and store the initial values at t=0
            sens_state, sens_cov = sensorModel.compute_EKF(q[0:3], obs)
        else: #move the simulators one step forward and compute the sensor model
            #dog follows velocity command
            q,v = dog.update()

            # drones update either with no commands or with a target drone and position
            if hold_and_fly:
                obs = drones.update(move_drone=next_drone, target_pos=INIT_drones[next_drone,:] + herm.eval(flight_time))
                flight_time += 1/SIM_FREQ
                if flight_time >= 2:
                    hold_and_fly = False
                    dog.set_cmd(B1_vel, B1_w)
                    sensorModel.reset_anchors(obs)
                    next_drone+= 1
                    if next_drone == num_drones:
                        next_drone = 0
                    flight_time = 0
                    sens_state, sens_cov = sensorModel.compute_EKF(q[0:3], obs)
                    print(sens_state, '\n \n', sens_cov, '\n \n')
            else:
                obs = drones.update()
                sens_state, sens_cov = sensorModel.compute_EKF(q[0:3], obs)

        # check to determine if the robot is currently moving or waiting for the drones to land
        if not hold_and_fly:
            #Store the pybullet GT world frame, EKF state estimation and state covariance for the new time step
            GT_states.append(np.append(q[:3], v[:3]))
            est_states.append(sens_state)
            cov.append(sens_cov.diagonal())
            counter +=1
            if delay < 25:
                delay +=25
            else:
                if sens_cov[0,0] > .145:
                    dog.set_cmd(np.zeros(3),0)
                    hold_and_fly = True
                    delay = 0
        
        
        
        
            
        
        
        

       
    np.savez('results_out', GT_states, est_states, cov)
    print('yay') 
    drones.close()
    


if __name__ == "__main__":
    run()