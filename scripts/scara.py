from math import *
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

        H_01 = np.array([[cos(th1), -sin(th1), 0, l2*cos(th1)],
                         [sin(th1), cos(th1), 0, l2*sin(th1)],
                         [0, 0, 1, l1],
                         [0, 0, 0, 1]
                         ])
        
        H_12 = np.array([[cos(th2), -sin(th2), 0, l4*cos(th2)],
                         [sin(th2), cos(th2), 0, l4*sin(th2)],
                         [0, 0, 1, l3-l5],
                         [0, 0, 0, 1]])

        H_23 = np.array([[1, 0, 0, 0],
                         [0, cos(pi), -sin(pi), 0],
                         [0, sin(pi), cos(pi), d3],
                         [0, 0, 0, 1]
                        ])

        H_LIST = [H_01, H_12, H_23]
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
