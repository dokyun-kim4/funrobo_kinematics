from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate
)


class TwoDOFRobot(TwoDOFRobotTemplate):
    def __init__(self):
        super().__init__()


    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        
        th1, th2 = curr_joint_values[0], curr_joint_values[1]
        l1, l2 = self.l1, self.l2

        H0_1 = np.array([[cos(th1), -sin(th1), 0, l1*cos(th1)],
                         [sin(th1), cos(th1), 0, l1*sin(th1)],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]
                        )

        H1_2 = np.array([[cos(th2), -sin(th2), 0, l2*cos(th2)],
                         [sin(th2), cos(th2), 0, l2*sin(th2)],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]
                        )
        
        Hlist = [H0_1, H1_2]

        # Calculate EE position and rotation
        H_ee = H0_1@H1_2  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist

    def calc_inverse_kinematics(self, ee: ut.EndEffector, joint_values: list[float], soln: int = 0):
        l1, l2 = self.l1, self.l2
        xd, yd = ee.x, ee.y
        L = sqrt(xd**2 + yd**2)
        B = np.arccos((l1**2 + l2**2 - L**2)/(2*l1*l2))
        th2 = pi - B if soln == 0 else -1*(pi - B)

        alpha = np.arctan2(l2*sin(th2), l1+l2*cos(th2))
        gamma = np.arctan2(yd,xd)
        th1 = gamma - alpha

        return [th1, th2]


if __name__ == "__main__":
    model = TwoDOFRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
