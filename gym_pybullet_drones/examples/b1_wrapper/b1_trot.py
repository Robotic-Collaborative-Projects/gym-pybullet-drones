## Contains b1 12 gait params
## Author : Avadesh Meduri
## Date : 11/7/22

import numpy as np
from b1_wrapper.weight_abstract import BiconvexMotionParams
from robot_properties_b1.config import B1Config

pin_robot = B1Config.buildRobotWrapper()
urdf_path = B1Config.urdf_path

#### Trot #########################################
trot = BiconvexMotionParams("b1", "Trot")

# Cnt
trot.gait_period = 0.5
trot.stance_percent = [0.6, 0.6, 0.6, 0.6]
trot.gait_dt = 0.05
trot.phase_offset = [0.0, 0.5, 0.5, 0.0]

# IK
trot.state_wt = np.array([0., 0, 10] + [5000, 5000, 1000] + [8e2] * (pin_robot.model.nv - 6) \
                        + [0.00] * 3 + [100, 100, 100] + [5e0] *(pin_robot.model.nv - 6))

trot.ctrl_wt = [0, 0, 1000] + [5e2, 5e2, 5e2] + [10.0] *(pin_robot.model.nv - 6)

trot.swing_wt = [1e5, 1e4]
trot.cent_wt = [3*[0*5e+1,], 6*[5e+1,]]
trot.step_ht = 0.15
trot.nom_ht = 0.4
trot.reg_wt = [5e-2, 1e-5]

# Dyn 
trot.W_X =        np.array([1e-5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+4, 1e+4, 1e4])
trot.W_X_ter = 10*np.array([1e+5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+5, 1e+5, 1e+5])
trot.W_F = np.array(4*[1e+2, 1e+2, 1e+2])
trot.rho = 5e+4
trot.ori_correction = [0.3, 0.9, 0.3]
trot.gait_horizon = 1.0
trot.kp = 150.0
trot.kd = 10.0
