
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
SIM_DURATION = 60
SIM_FREQ = 1000
dt = .001

INIT_B1_pos = np.array([0,0,.4])
B1_vel = np.array([.3,0,0])
B1_w = 0.00

#initialize the drone swarm positions and rotations

num_drones = 6
INIT_drones_rots = np.array([[0, 0,  0] for i in range(num_drones)])
INIT_drones = np.zeros((num_drones,3))
ctr = 0
for i in range(num_drones):
 
    if i % 2 == 0:
        INIT_drones[i,:] = np.array([[-1+ctr,1,.0]])
    else: 
        INIT_drones[i,:] = np.array([[-1+ctr,-1,.0]])
        ctr +=1

#define a trajectory forward by the split distance * the number of drones.
traj = trajectory.Trajectory(milestones=[[0,0,0], [ctr/2, 0, .4], [ctr, 0, 0]])
herm = trajectory.HermiteTrajectory()
herm.makeSpline(traj)

GT_states = []
est_states = []
est_cov = []

drones_GT = []
drones_est_state = []
drones_cov = []


def run():    
    # Set up the Drones
    drones = DroneEnvWrapper(INIT_XYZS=INIT_drones, num_drones=num_drones, INIT_RPYS=INIT_drones_rots, simulation_freq_hz=SIM_FREQ, duration_sec=SIM_DURATION)
    obs = drones.get_obs_no_update()
    next_drone = 0
    flight_time = 0
    
    #Set up the dog
    dog = B1Wrapper(b1_params, initial_position=INIT_B1_pos)
    dog.set_cmd(B1_vel,B1_w)
    dogModel = dogEKFModel(INIT_B1_pos, np.zeros(3), INIT_drones, dt)
    q, v = dog.get_b1_state()
    hold_and_fly = False
    delay = 0
    dog_measurement_count= 0
    save_drone = True
    for i in range(0, int(SIM_DURATION*SIM_FREQ)):
            
        if i == 0: #compute and store the initial values at t=0
            est_state, cov = dogModel.compute_EKF(q[0:3], obs)
            
            est_states.append(est_state)
            est_cov.append(cov)
            GT_dog = np.append(q[:3],v[:3])
            GT_states.append(GT_dog)
            
        else: #move the simulators one step forward and compute the sensor model
            #dog follows velocity command
            q,v = dog.update()

            # drones update either with no commands or with a target drone and position
            if hold_and_fly:
                # initialize the current drone ekf with the current position
                if flight_time == 0:
                    droneModel = droneEKFModel(next_drone, obs, dt)
                    drone_est_state, drone_cov = droneModel.compute_EKF(obs)
                    drones_GT.append(np.append(obs[str(next_drone)]['state'][:3],obs[str(next_drone)]['state'][10:13]))
                    drones_est_state.append(drone_est_state)
                    drones_cov.append(drone_cov)
                if flight_time < 2:
                    obs = drones.update(move_drone=next_drone, target_pos=INIT_drones[next_drone,:] + herm.eval(flight_time))
                else:
                    obs = drones.update(move_drone=next_drone, target_pos=INIT_drones[next_drone,:] + herm.eval(2))
                    
                if save_drone == True:
                    drone_est_state, drone_cov = droneModel.compute_EKF(obs)
                    drones_GT.append(np.append(obs[str(next_drone)]['state'][:3],obs[str(next_drone)]['state'][10:13]))
                    drones_est_state.append(drone_est_state)
                    drones_cov.append(drone_cov)
                flight_time += dt   

                if flight_time > 2.3:
                    hold_and_fly = False
                    save_drone = False
                    dogModel.reset_anchors(obs)
                    next_drone += 1
                    if next_drone == num_drones:
                        next_drone = 0
                    dog.set_cmd(B1_vel, B1_w)
                    flight_time = 0
                    #print(sens_state, '\n \n', sens_cov, '\n \n')
            else:
                obs = drones.update()
                # Store current step information
                est_state, cov = dogModel.compute_EKF(q[0:3], obs)
                est_states.append(est_state)
                est_cov.append(cov)
                GT_dog = np.append(q[:3], v[:3])
                GT_states.append(GT_dog)
        
            
        # check to determine if the robot is currently moving or waiting for the drones to land
            if not hold_and_fly:
                if delay < 100:
                    delay +=1
            #Store the pybullet GT world frame, EKF state estimation and state covariance for the new time step
                elif cov[0,0] > .03:
                    dog.set_cmd(np.zeros(3),0)
                    hold_and_fly = True
                    
                    delay = 0
        # store the current EKF estimates of the system

        p.stepSimulation()
            

       
    np.savez('dog_results_out', GT_states, est_states, est_cov)
    np.savez('drone_results_out', drones_GT, drones_est_state, drones_cov)
    print('yay') 
    drones.close()
    


if __name__ == "__main__":
    run()