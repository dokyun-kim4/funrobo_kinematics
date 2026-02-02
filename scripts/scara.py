from math import *
import numpy as np
from funrobo_kinematics.core import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import ScaraRobotTemplate


class TwoDOFRobot(ScaraRobotTemplate):
    def __init__(self):
        super().__init__()
    
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        th1, th2, th3 = curr_joint_values[0], curr_joint_values[1], curr_joint_values[2]
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        H_01 = np.array([[cos(th1), -sin(th1), 0, l2],
                         [sin(th1), cos(th1), 0, 0],
                         [0, 0, 1, l1],
                         [0, 0, 0, 1]
                         ])
        
        H_12 = np.array([[cos(th2), -sin(th2), 0, l4],
                         [sin(th2), cos(th2), 0, 0],
                         [0, 0, 1, l3-l5],
                         [0, 0, 0, 1]])

        H_23 = np.array([[1, 0, 0, 0],
                         [0, cos(th3), -sin(th3), 0],
                         [0, sin(th3), cos(th3), 0],
                         [0, 0, 0, 1]
                        ])



if __name__ == "__main__":
    
    model = TwoDOFRobot()
    
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
