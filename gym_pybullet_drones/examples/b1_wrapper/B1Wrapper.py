import numpy as np
import pybullet
import pinocchio as pin
from robot_properties_b1.config import B1Config
from robot_properties_b1.b1wrapper import B1Robot
from b1_wrapper.mpc.anymal_cyclic_gen import AnymalMpcGaitGen
from b1_wrapper.inverse_dynamic_controller import InverseDynamicsController


class B1Wrapper():
    
    def __init__(self, params, initial_position = np.array([0, 0, 0.4])):
        self.params = params       
        #simulation time and counter variables 
        self.sim_t = 0
        self.index = 0
        self.pln_ctr = 0
        self.o = 0
        self.lag = int(params["update_time"]/params["sim_dt"])
        self.f_arr = ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
        ## robot config and init
        self.pin_robot = B1Config.buildRobotWrapper()
        self.urdf_path = B1Config.urdf_path
        self.n_eff = 4
        #Generate the initial configuration of the robot
        q0 = np.array(B1Config.initial_configuration)
        q0[0:3] = initial_position
        v0 = pin.utils.zero(self.pin_robot.model.nv)
        x0 = np.concatenate([q0, pin.utils.zero(self.pin_robot.model.nv)])
        #Instantiate the robot in the simulator
        self.robot = B1Robot()
        self.robot.reset_state(q0, v0)
        #Bicon Planner
        self.gg = AnymalMpcGaitGen(self.pin_robot, self.urdf_path, x0, params['plan_freq'], q0, None)
        self.gg.update_gait_params(params['gait_params'], self.sim_t)
        #Inverse Dynamics Controller
        self.robot_id_ctrl = InverseDynamicsController(self.pin_robot, self.f_arr)
        self.robot_id_ctrl.set_gains(params['gait_params'].kp, params['gait_params'].kd)
        #Velocity Commands to the robot
        self.v_des = np.zeros((3,))
        self.w_des = 0

    
    def update(self, desired_state=np.zeros((2,))):
        # Get the robot states
        q, v = self.robot.get_state()
        
        # Position Controller
        error = desired_state-q[0:2]    
        if abs(error[0]) > .15:
            self.v_des[0] = np.sign(error[0])*.35
        else:
            self.v_des[0] = np.sign(error[0])*.05

        if abs(error[1]) > .15:
            self.v_des[1] = np.sign(error[1])*.2
        else:
            self.v_des[1] = np.sign(error[1])*.05

        # BiconMP velocity Controller
        q[3:7] = self.quaternion_workaround(q) #Reset yaw to zero
        contact_configuration = self.robot.get_force()[0]

        if self.o == int(100*(self.params['plan_freq']/self.params['sim_dt'])):
            self.gg.update_gait_params(self.params['gait_params'], self.params['sim_dt'])
            self.robot_id_ctrl.set_gains(self.params['gait_params'].kp, self.params['gait_params'].kd)

        if self.pln_ctr == 0:        
            self.xs_plan, self.us_plan, self.f_plan = self.gg.optimize(q, v, np.round(self.sim_t,3), self.v_des, self.w_des)

        # first loop assume that trajectory is planned
        if self.o < int(self.params['plan_freq']/self.params['sim_dt']) - 1:
            self.xs = self.xs_plan
            self.us = self.us_plan
            self.f = self.f_plan

        # second loop onwards lag is taken into account
        elif self.pln_ctr == self.lag and self.o > int(self.params['plan_freq']/self.params['sim_dt'])-1:
            self.lag = 0
            self.xs = self.xs_plan[self.lag:]
            self.us = self.us_plan[self.lag:]
            self.f = self.f_plan[self.lag:]
            self.index = 0

        tau = self.robot_id_ctrl.id_joint_torques(q, v, self.xs[self.index][:self.pin_robot.model.nq].copy(), self.xs[self.index][self.pin_robot.model.nq:].copy()\
                                    , self.us[self.index], self.f[self.index], contact_configuration)
        self.robot.send_joint_command(tau)
        self.sim_t += self.params['sim_dt']

        # pln_ctr becomes zero for every MPC collocation. In this project, this would be every 50ms
        self.pln_ctr = int((self.pln_ctr + 1)%(self.params['plan_freq']/self.params['sim_dt']))
        self.o +=1
        self.index +=1
        return self.robot.get_state()

    def set_cmd(self, v_des, w_des):
        self.v_des = v_des
        self.w_des = w_des

    def reset(self):
        self.sim_t = 0
        self.index = 0
        self.pln_ctr = 0
        self.o = 0

    def quaternion_workaround(self, q):
        R = pin.Quaternion(np.array(q[3:7])).toRotationMatrix()
        rpy_vector = pin.rpy.matrixToRpy(R)
        rpy_vector[2] = 0.0
        fake_quat = pin.Quaternion(pin.rpy.rpyToMatrix(rpy_vector))
        return np.array([fake_quat[0], fake_quat[1], fake_quat[2], fake_quat[3]])
    
    def start_recording(self, file_name):
        self.file_name = file_name
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)

    def stop_recording(self):
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)