from math import *
import pandas as pd
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import ScaraRobotTemplate


class ScaraRobot(ScaraRobotTemplate):
    def __init__(self):
        super().__init__()
    
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        th1, th2, d3 = curr_joint_values[0], curr_joint_values[1], curr_joint_values[2]
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        DH_TABLE = pd.DataFrame(columns=["joint", "theta", "d", "a", "alpha"])
        DH_TABLE.joint = [i+1 for i in range(len(curr_joint_values))] # note joint indexes from 1
        DH_TABLE.theta = [th1, th2, 0]
        DH_TABLE.d = [l1, l3-l5, -d3]
        DH_TABLE.a = [l2, l4, 0]
        DH_TABLE.alpha = [0, 0, pi]

        def get_H_i(i: int) -> np.ndarray:
            curr_joint_params = DH_TABLE[DH_TABLE.joint == i].values[0][1:] # skip joint name from values
            theta, d, a, alpha = curr_joint_params
            return np.array([[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
                             [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                             [0, sin(alpha), cos(alpha), d],
                             [0, 0, 0, 1]
                             ])

        H_LIST = [get_H_i(i+1) for i in range(len(curr_joint_values))]
        H_01, H_12, H_23 = H_LIST
       
        H_EE = H_01@H_12@H_23  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_EE @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_EE[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, H_LIST


if __name__ == "__main__":
    
    model = ScaraRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
