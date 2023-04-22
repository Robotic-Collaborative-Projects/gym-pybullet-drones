"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
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

# import B1 related files
from b1_wrapper.B1Wrapper import B1Wrapper
from b1_wrapper.b1_trot import trot
from bullet_utils.env import BulletEnvWithGround
from klampt.model import trajectory


b1_params = {
    "plan_freq":     0.05,
    "update_time":   0.02,
    "sim_dt":       .001,
    "gait_params":   trot
}

# print(env.CLIENT)
# p.setGravity(0,0,-9.8)
# p.setPhysicsEngineParameter(fixedTimeStep=0.001, numSubSteps=1)
# env = BulletEnvWithGround()
# dog = B1Wrapper(b1_params, initial_position=np.array([0,0,0.4]))

# while True:
#     dog.update()
#     env.step()

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
DEFAULT_DURATION_SEC = 4
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def genTraj(initial_pos, desired_pos, extra_pos=None, milestone_times=None):
    """
    generates the trajectory for a single robot. extra should be a list of desired positions along the way. 
    Outputs trajectory object which can be evaluated using .eval
    """
    milestones = initial_pos.append(extra_pos.append(desired_pos))

    traj = trajectory.HermiteTrajectory(milestone_times,milestones), 

    return traj

def sensorModel(params=None, b1_state=None, drones_obs=None):
    output_dict = {}
    # Dog wrt Drones 1-4
    for drone_id in drones_obs.keys():
        output_dict[f'B1D{drone_id}'] = np.linalg.norm(b1_state-drones_obs[drone_id]['state'][0:3])
    # output_dict['B1D2'] = np.linalg.norm(b1_state-drones_obs['1']['state'][0:3])
    # output_dict['B1D3'] = np.linalg.norm(b1_state-drones_obs['2']['state'][0:3])
    # output_dict['B1D4'] = np.linalg.norm(b1_state-drones_obs['3']['state'][0:3])
    for i in range(len(drones_obs.keys())):
        for j in range(len(drones_obs.keys())):
            if i!=j:
                output_dict[f'D{i}D{j}'] = np.linalg.norm(drones_obs[f'{i}']['state'][0:3]-drones_obs[f'{j}']['state'][0:3])

    #Drone 1 wrt Drones 3,4
    # output_dict['D1D2'] = np.linalg.norm(drones_obs['0']['state'][0:3]-drones_obs['1']['state'][0:3])
    # output_dict['D1D3'] = np.linalg.norm(drones_obs['0']['state'][0:3]-drones_obs['2']['state'][0:3])
    # output_dict['D1D4'] = np.linalg.norm(drones_obs['0']['state'][0:3]-drones_obs['3']['state'][0:3])
    # #Drone 2 wrt Drones 3,4
    # output_dict['D2D3'] = np.linalg.norm(drones_obs['1']['state'][0:3]-drones_obs['2']['state'][0:3])
    # output_dict['D2D4'] = np.linalg.norm(drones_obs['1']['state'][0:3]-drones_obs['3']['state'][0:3])
    # #Drone 3 wrt Drone 4
    # output_dict['D3D4'] = np.linalg.norm(drones_obs['2']['state'][0:3]-drones_obs['3']['state'][0:3])
    return output_dict

def run(drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        vision=DEFAULT_VISION,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    INIT_B1 = np.array([0,0,.5])
    INIT_XYZS = np.zeros((num_drones,3))
    pair_counter = 0
    for i in range(num_drones):
        if i%2==0:
            INIT_XYZS[i,:] = np.array([[-.5+pair_counter,1,.0]])
        else:
            INIT_XYZS[i,:] = np.array([[-.5+pair_counter,-1,.0]])
            pair_counter+=1
    
    INIT_RPYS = np.array([[0, 0,  0] for i in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1

    #### Initialize a circular trajectory ######################
    NUM_WP = control_freq_hz*duration_sec
    TARGET_B1 = np.array([4,0])

    #### Create the environment with or without video capture ##
    env = CtrlAviary(drone_model=drone,
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
        
    #### Obtain the PyBullet Client ID from the environment ####
    # PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    elif drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(num_drones)}
    dog = B1Wrapper(b1_params)

    q,v = dog.update(TARGET_B1)
    loop = 1

    traj = trajectory.Trajectory(milestones=[[0,0,0],[pair_counter/2,0,.5], [pair_counter,0,0]])
    while(loop<=pair_counter):
        START = time.time()
        counter = 0
        while(q[0]<loop):
            print(q[0])
            q,v = dog.update(TARGET_B1)
            obs, reward, done, info = env.step(action)
            if counter%env.SIM_FREQ == 0:
                env.render()
                if gui:
                    sync(counter, START, env.TIMESTEP)
            counter +=1
        

        for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

            #### Make it rain rubber ducks #############################
            # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)
            q,v = dog.update(np.array([loop,0]))
            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)
            mydict = sensorModel(b1_state=q[0:3],drones_obs=obs)
            # print(mydict)
            # print(obs['0']['state'])
            print(traj.eval(2*i/(int(duration_sec*env.SIM_FREQ))))
            #### Compute control at the desired frequency ##############
            if i%CTRL_EVERY_N_STEPS == 0:

                #### Compute control for the current way point #############
                for j in range(0,num_drones):
                    if j <=loop and j >= loop-1:
                        action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                        state=obs[str(j)]["state"],
                                                                        target_pos=INIT_XYZS[j,:]+traj.eval(2*i/(int(duration_sec*env.SIM_FREQ))),
                                                                        target_rpy=INIT_RPYS[j, :],
                                                                        )
                    else:
                        action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                        state=obs[str(j)]["state"],
                                                                        target_pos=INIT_XYZS[j,:],
                                                                        target_vel=[0,0,0],
                                                                        target_rpy=INIT_RPYS[j, :],
                                                                        )

            #### Log the simulation ####################################
            for j in range(0,num_drones):
                if j <=loop and j >= loop-1:
                    logger.log(drone=j,
                        timestamp=i/env.SIM_FREQ,
                        state=obs[str(j)]["state"],
                        control=np.hstack([INIT_XYZS[j,:]+traj.eval(2*i/(int(duration_sec*env.SIM_FREQ))), INIT_RPYS[j, :], np.zeros(6)])
                        # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                        )
                else: 
                    logger.log(drone=j,
                        timestamp=i/env.SIM_FREQ,
                        state=obs[str(j)]["state"],
                        control=np.hstack([INIT_XYZS[j,:], INIT_RPYS[j, :], np.zeros(6)])
                        # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                        )

            #### Printout ##############################################
            if i%env.SIM_FREQ == 0:
                env.render()
                #### Print matrices with the images captured by each drone #
                if vision:
                    for j in range(num_drones):
                        print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                            obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
                            obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
                            )

            #### Sync the simulation ###################################
            if gui:
                sync(i, START, env.TIMESTEP)
        loop +=1

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=DEFAULT_VISION,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=DEFAULT_AGGREGATE,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))