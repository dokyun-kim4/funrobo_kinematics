from math import *
import pandas as pd
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate

class HiWonder5DOF(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
    
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        th1,th2,th3,th4,th5 = self.joint_values

        # order or variables: theta, d, a, alpha
        DH = np.array([[th1, l1, 0, -pi/2],
                       [th2-pi/2, 0, l2, pi],
                       [th3, 0, l3, pi],
                       [pi/2+th4, 0, 0, pi/2],
                       [th5, l4+l5, 0, 0]])
        H_LIST = [ut.dh_to_matrix(DH[i]) for i in range(len(curr_joint_values))]
        H_01, H_12, H_23, H_34, H_45 = H_LIST
        H_EE = H_01@H_12@H_23@H_34@H_45  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_EE @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_EE[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, H_LIST


if __name__ == "__main__":
    
    model = HiWonder5DOF()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
