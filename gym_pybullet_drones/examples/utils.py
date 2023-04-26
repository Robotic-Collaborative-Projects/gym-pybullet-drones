from klampt.model import trajectory
from scipy.spatial.transform import Rotation as rot
import numpy as np
import pybullet as p

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

    for i in range(len(drones_obs.keys())):
        for j in range(len(drones_obs.keys())):
            if i!=j:
                output_dict[f'D{i}D{j}'] = np.linalg.norm(drones_obs[f'{i}']['state'][0:3]-drones_obs[f'{j}']['state'][0:3])
    return output_dict


class DronesSensorModel():
    def __init__(self, num_robots, initial_xyzs, initial_rpys):
        """
        initial_offset is a num_robots x 3 x 2 array, representing the n by xyz by rpy for n robots in the system. For consistency, quadrupeds will be labeled as the first "dog" arrays, and the drones will be labeled as the following "bird" arrays
        sets up the system offset which is a num_robots x 4 x 4, array representing the inverse transformation of 
        
        """
        ##### Main Parameters
        self.num_robots = num_robots
        # offsets of shape: (num_robotsx4x4)
        self.array_of_robot_origin_transformations_expressed_in_the_world_frame = self.xyz_rpy_to_homogeneous_transform(initial_xyzs, initial_rpys)
        self.previous_poses = None

        ##### Noise Parameters

    def update(self, obs):
        '''
        This function aims to create internal odometry poses and return the noisy version of the obs
        '''
        self.latest_gt = osb
        if self.previous_poses is None:
            # generate identity poses as the observation that we return to the controller

        else:
            # we want to integrate the odometry starting from our initial offset
        
    
    def reset_offset(self, new_offset):
        self.system_offset = self.xyz_rpy_to_homogeneous_transform(new_offset[0,:], new_offset[1,:])

    def xyz_rpy_to_homogeneous_transform(self,xyzs,rpys):
        """
        converts two num_robots x 3 arrays for the xyz position and rpy into an num_robots 4x4 homogeneous transformation, R,t, where the last row is [0,0,0,1
        """
        Ts_out = np.zeros((self.num_robots,4,4))
        for i in range(self.num_robots):
            Ts_out[i,:3,:3] = rot.from_euler('zyx', rpys[i,:]).as_matrix() #double check the euler angles
            Ts_out[i,:3,3] = xyzs[i,:]
            Ts_out[i,3,3] = 1
        print(Ts_out)
        return Ts_out

    def process_new_odometry(self, curr_b1_state, curr_drones_obs):
        """
        processes the state from the drone observation, corrupts the information with noise, and 
        """
        return None
     
    def get_UWB_measurements(self, curr_b1_state, curr_drones_obs):
        output_dict = {}
        for drone_id in curr_drones_obs.keys():
            output_dict[f'B1D{drone_id}'] = np.linalg.norm(curr_b1_state-curr_drones_obs[drone_id]['state'][0:3])
        for i in range(len(curr_drones_obs.keys())):
            for j in range(len(curr_drones_obs.keys())):
                if i!=j:
                    output_dict[f'D{i}D{j}'] = np.linalg.norm(curr_drones_obs[f'{i}']['state'][0:3]-curr_drones_obs[f'{j}']['state'][0:3])

    def getImage(self, drone_idx):
        drone_gt_state = self.obs[f'{drone_idx}']


        return rgb_img, seg_img, depth_img
        