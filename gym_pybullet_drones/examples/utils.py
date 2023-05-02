from klampt.model import trajectory
from scipy.spatial.transform import Rotation as rot
import numpy as np
import pybullet as p


class dogEKFModel():
    def __init__(self, robot_initial_position, robot_initial_vel, anchor_points, dt):
        '''
        this is the dog EKF class model that takes in the initial position and velocity of the robot in the world frame (3,)
            the anchors (num_anchors,3) and dt, the timestep size of the filter.

            
        '''
        # store the anchor point positions and number of anchorpoints
        self.anchor_points = anchor_points
        self.num_anchors = np.shape(anchor_points)[0]
        
        # initialize the previous mean and covariance
        self.prev_state_mean = np.zeros((6,))
        self.prev_state_mean[:3] = robot_initial_position
        self.prev_state_mean[3:] = robot_initial_vel

        self.prev_state_cov = np.diag([.03,.03,.03,1,1,1])


        self.curr_state_mean = np.zeros((6,))
        self.curr_state_covar = np.zeros((6,6))
        
        # define the process and measurement noise
        self.measurement_noise = .3*np.eye(self.num_anchors)
        self.process_noise = np.diag([1,1,1,100,100,100])

        # store the A matrix for the linear process model
        self.A = np.eye(6)
        self.A[:3,3:] = np.eye(3)*dt

        # print(self.A, '\n \n')
        

    def compute_EKF(self, new_B1_pos, new_obs):
        '''
        this takes in the
        '''
        self.prediction_step()
        self.update_step(new_B1_pos, new_obs)
        self.prev_state_mean = self.curr_state_mean
        self.prev_state_cov = self.curr_state_covar
        return self.curr_state_mean, np.diag(self.curr_state_covar)


    def prediction_step(self):
        self.curr_est_mean = self.A @ self.prev_state_mean 
        self.curr_est_covar = self.A @ self.prev_state_cov @ np.transpose(self.A) + self.process_noise

        # print(self.curr_est_mean, '\n \n', self.curr_est_covar, '\n \n')
        
    
    def update_step(self, new_pos, obs):
        
        # compute the GT measurement vector
        zt, _ = self.dogSensorModel(new_pos, obs)
        # compute the measurement model with the estimated state, evaluate the jacobian of the measurement model at the estimated state
        z_est, Ct = self.dogSensorModel(self.curr_est_mean[:3], obs)

        # print(zt, '\n', '\n', z_est, '\n', '\n', Ct, '\n', '\n')
        Kt = self.curr_est_covar @ np.transpose(Ct) @ np.linalg.inv(Ct @ self.curr_est_covar @ np.transpose(Ct) + self.measurement_noise)
        # compute the curren state mean
        self.curr_state_mean = self.curr_est_mean + Kt @ (zt-z_est)
        self.curr_state_covar = self.curr_est_covar - Kt @ Ct @ self.curr_est_covar
        # print(self.curr_state_mean, '\n \n', self.curr_state_covar, '\n \n')


    def compute_jacobian_one_robot(self, robot, other_robot, denom):
        output = np.zeros((3,))
        output[0] = (robot[0] - other_robot[0]) / denom
        output[1] = (robot[1] - other_robot[1]) / denom
        output[2] = (robot[2] - other_robot[2]) / denom
        return output

    def dogSensorModel(self, b1_pos=None, drones_obs=None):
        output_array = np.zeros((len(drones_obs.keys())))
        Ct = np.zeros((self.num_anchors,6))
        # Dog wrt Drones 1-4
        for idx in range(len(drones_obs.keys())):
            drone_pos = drones_obs[str(idx)]['state'][0:3]
            dist = np.linalg.norm(b1_pos-drone_pos)
            output_array[idx] = dist
            Ct[idx,:3] = self.compute_jacobian_one_robot(b1_pos, drone_pos, dist)[:]

        return output_array, Ct    
    
    def reset_anchors(self, obs):
        for i in range(len(obs.keys())):
            self.anchor_points[i,:] = obs[str(i)]['state'][:3]
            # print(self.anchor_points)
        

class droneEKFModel():
    def __init__(self, drone_num, initial_obs, dt):
        # store the anchor point positions
        self.num_anchors = len(initial_obs.keys())-1
        self.drone_num = drone_num
        self.anchor_points = np.zeros((self.num_anchors,3))
        idx = 0
        for key in initial_obs.keys():
            if int(key) != self.drone_num:
                self.anchor_points[idx,:] = initial_obs[key]['state'][:3]
            else:
                self.prev_state_mean = np.zeros((6,))
                self.prev_state_mean[:3] = initial_obs[key]['state'][:3]
                self.prev_state_mean[3:] = initial_obs[key]['state'][10:13]

        
        # initialize the previous mean and covariance
        
        self.prev_state_cov = np.eye(6)*.1
        self.curr_state_mean = np.zeros((6,))
        self.curr_state_covar = np.zeros((6,6))
        
        # define the process and measurement noise
        
        self.process_noise = np.diag([1,1,1,10,10,10])

        self.measurement_noise = .1*np.eye(self.num_anchors)

        # store the A matrix for the linear process model
        self.A = np.eye(6)
        self.A[:3,3:] = np.eye(3)*dt

    def prediction_step(self):
        self.curr_est_mean = self.A @ self.prev_state_mean 
        self.curr_est_covar = self.A @ self.prev_state_cov @ np.transpose(self.A) + self.process_noise

        # print(self.curr_est_mean, '\n \n', self.curr_est_covar, '\n \n')
        
    
    def update_step(self, new_obs):
        
        # compute the GT measurement vector
        zt, _ = self.droneSensorModel(new_obs)
        # compute the measurement model with the estimated state, evaluate the jacobian of the measurement model at the estimated state
        z_est, Ct = self.droneSensorModel(new_obs, self.curr_est_mean[:3])

        # print(zt, '\n', '\n', z_est, '\n', '\n', Ct, '\n', '\n')
        Kt = self.curr_est_covar @ np.transpose(Ct) @ np.linalg.inv(Ct @ self.curr_est_covar @ np.transpose(Ct) + self.measurement_noise)
        # compute the curren state mean
        self.curr_state_mean = self.curr_est_mean + Kt @ (zt-z_est)
        self.curr_state_covar = self.curr_est_covar - Kt @ Ct @ self.curr_est_covar
        # print(self.curr_state_mean, '\n \n', self.curr_state_covar, '\n \n')

    def compute_jacobian_one_robot(self, robot, other_robot, denom):
        output = np.zeros((3,))
        output[0] = (robot[0] - other_robot[0]) / denom
        output[1] = (robot[1] - other_robot[1]) / denom
        output[2] = (robot[2] - other_robot[2]) / denom
        return output

    def droneSensorModel(self, drones_obs, mean=None):
        output_array = np.zeros((len(drones_obs.keys())))
        Ct = np.zeros((self.num_drones,6))
        # Drone x  wrt Drones 1-4
        if mean is None:
            drone_pos = drones_obs[str(self.drone_num)]['state'][:3]
        else:
            drone_pos = mean
        idx = 0
        for j in range(len(drones_obs.keys())):
            if j != self.drone_num:
                other_drone_pos = drones_obs[str(j)]['state'][:3]
                dist = np.linalg.norm(drone_pos-other_drone_pos)
                output_array[idx] = dist
                Ct[idx,:3] = self.compute_jacobian_one_robot(drone_pos, other_drone_pos, dist)[:]

        return output_array, Ct 


# class DronesSensorModel():
#     def __init__(self, num_robots, initial_xyzs, initial_rpys):
#         """
#         initial_offset is a num_robots x 3 x 2 array, representing the n by xyz by rpy for n robots in the system. For consistency, quadrupeds will be labeled as the first "dog" arrays, and the drones will be labeled as the following "bird" arrays
#         sets up the system offset which is a num_robots x 4 x 4, array representing the inverse transformation of 
        
#         """
#         ##### Main Parameters
#         # useful for later
#         self.num_drones = 6
#         self.num_B1 = 0
#         self.num_robots = self.num_drones+self.num_B1

#         #initialize storing the the ground truth poses as num_robots x 4x4 and num_robots x 6x6 
#         self.previous_GT_poses = None
#         #initialize storing the noisy robot frame 
#         self.previous_odometry = None
        

#         # offsets of shape: (num_robotsx4x4)
#         self.w4x4r, w6x6r, r4x4w, r6x6w  = self.xyz_rpy_to_homogeneous_transforms(initial_xyzs, initial_rpys)
#         self.robot_prev_in_robot_list = None

#         ##### Noise Parameters

#     def update(self, obs, formation):
#         '''
#         This main function processes a new observation. The input is the current robot frame expressed in the pybullet world frame.
#         The goal of this function is to execute the following steps:
#             1. Unpack each robot bundle from the current timestep into: position, orientation(rpy), velocity, ang velocity (world frame),
#                rgb, depth, segmentation images for each robot in the system.
#             2. Move the state into the robot frame by calling a transform state function.
#             3. Measure the GT distance between each robot for the UWB model
#             4. Apply some noise to the system's state to simulate drift from the desired trajectory.
#             5. Apply some noise to the UWB model
#             6. Return the estimated system state and desired control
#         '''
#         #pos[0:3], rpy[7:10], lin vel[10:13], ang vel[13:16]

#         state_size = 3+3+3+3
#         self.robot_curr_state_in_world = np.hstack([obs[i]['state'][:state_size] for i in range(self.num_robots)])
#         if self.previous_poses is None:
#             # generate identity poses as the observation that we return to the controller
#             self.previous_poses = [np.vstack(np.eye(4) for i in range(self.num_robots))]
#         else:
#             # we want to integrate the odometry starting from our initial offset
#             self.previous_GT_poses.append()


#     def xyz_rpy_to_homogeneous_transforms(self,xyzs,rpys):
#         """
#         converts two num_robots x 3 arrays for the xyz position and rpy into an num_robots 4x4 homogeneous transformation, R,t, where the last row is [0,0,0,1
#         """
#         T_out = np.zeros((self.num_robots,4,4))
#         Tv_out = np.zeros((self.num_robots, 6,6))
#         T_out_inv = np.zeros_like(T_out)
#         Tv_out_inv = np.zeros_like(Tv_out)
#         for i in range(self.num_robots):
#             R = rot.from_euler('zyx', rpys[i,:]).as_matrix()
#             t = xyzs[i,:]
#             T_out[i,:,:] = np.vstack((np.hstack((R, t)),np.array([0,0,0,1]))) #double check the euler angles
#             T_out_inv[i,:,:] = np.linalg.inv( T_out[i,:,:] )
#             S = np.array( [ [ 0, -t[2], t[1] ],
#                             [ t[2], 0, -t[0] ],
#                             [ -t[1], t[0], 0 ] ] )
#             Tv_out[i,:,:] = np.vstack( ( np.hstack( ( R, np.zeros((3,3))  ) ), 
#                                          np.hstack( ( -np.matmul(S, R), R ) ) ) )
#             Tv_out_inv[i,:,:] = np.linalg.inv( Tv_out[i,:,:] )


#         return T_out, Tv_out, T_out_inv, Tv_out_inv


#     def get_UWB_measurements(self, curr_b1_state, curr_drones_obs):
#         output_dict = {}
#         for drone_id in curr_drones_obs.keys():
#             output_dict[f'B1D{drone_id}'] = np.linalg.norm(curr_b1_state-curr_drones_obs[drone_id]['state'][0:3])
#         for i in range(len(curr_drones_obs.keys())):
#             for j in range(len(curr_drones_obs.keys())):
#                 if i!=j:
#                     output_dict[f'D{i}D{j}'] = np.linalg.norm(curr_drones_obs[f'{i}']['state'][0:3]-curr_drones_obs[f'{j}']['state'][0:3])

#     def getImage(self, drone_idx):
#         drone_gt_state = self.obs[f'{drone_idx}']


#         return rgb_img, seg_img, depth_img
        