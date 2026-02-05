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

        ## Calculate H matrix for {0} -> {1}

        # Compute R and d separately
        R_01 = np.array([[cos(th1), -sin(th1), 0],
                         [sin(th1), cos(th1), 0],
                         [0, 0, 1]
                         ])
        
        d_01 = np.array([[l2*cos(th1)],
                         [l2*sin(th1)],
                         [l1]
                         ])
        
        # Combine R and d into H
        H_01 = np.eye(4)
        H_01[0:3, 0:3] = R_01
        print(H_01)
        H_01[0:3, 3:4] = d_01
        print(H_01)
        
        # Calculate H matrix for {1} -> {2}
        R_12 = np.array([[cos(th2), -sin(th2), 0],
                         [sin(th2), cos(th2), 0],
                         [0, 0, 1]
                         ])
        
        d_12 = np.array([[l4*cos(th2)],
                         [l4*sin(th2)],
                         [l3 - l5]
                         ])

        H_12 = np.eye(4)
        H_12[0:3, 0:3] = R_12
        H_12[0:3, 3:4] = d_12
        
        # Calculate H matrix for {2} -> {3}(End Effector)
        R_23 = np.array([[1, 0, 0],
                         [0, cos(pi), -sin(pi)],
                         [0, sin(pi), cos(pi)]
                         ])
        
        d_23 = np.array([[0],
                         [0],
                         [d3]
                         ])
        
        H_23 = np.eye(4)
        H_23[0:3, 0:3] = R_23
        H_23[0:3, 3:4] = d_23

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
