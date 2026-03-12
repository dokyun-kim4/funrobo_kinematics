from math import *
from typing import List, Tuple
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim # type: ignore
from funrobo_kinematics.core.arm_models import KinovaRobotTemplate


class Kinova6DOF(KinovaRobotTemplate):
    def __init__(self):
        super().__init__()
        self.vel_lower_lim = np.array([l[0] for l in self.joint_vel_limits])
        self.vel_upper_lim = np.array([l[1] for l in self.joint_vel_limits])
        self.joint_lower_lim = np.array([l[0] for l in self.joint_limits])
        self.joint_upper_lim = np.array([l[1] for l in self.joint_limits])
    
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        l1, l2, l3, l4, l5, l6, l7 = self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7
        th1, th2, th3, th4, th5, th6 = curr_joint_values if radians else np.rad2deg(curr_joint_values)

        # Note there is an extra frame {B} at the base, accounts for downward z axis at joint 1
        DH = np.array([[0, 0, 0, pi],
                       [th1, -l1-l2, 0, pi/2],
                       [-pi/2 + th2, 0, l3, pi],
                       [-pi/2 + th3, 0, 0, pi/2],
                       [th4, -l4-l5, 0, -pi/2],
                       [th5, 0, 0, pi/2],
                       [th6, -l6-l7, 0, pi]])

        H_LIST = [ut.dh_to_matrix(DH[i]) for i in range(len(curr_joint_values) + 1)]
        H_B0, H_01, H_12, H_23, H_34, H_45, H_56 = H_LIST
        H_EE = H_B0@H_01@H_12@H_23@H_34@H_45@H_56  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_EE @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_EE[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, H_LIST
    
    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt: float = 0.02):
        """
        Calculates the velocity kinematics for the robot based on the given end effector velocity input.

        Args:
            joint_values (list): The current joint values of the robot.
            vel (list): The velocity vector for the end effector [vx, vy, vz].
            dt (float): The time step for the velocity update.

        Returns:
            new_joint_values (list): The updated joint values after applying the velocity input.
        """
        new_joint_values = joint_values.copy()

        # move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]


        JV_INV = self.calc_inv_jacobian(new_joint_values)

        joint_vel = JV_INV @ vel
        
        joint_vel = np.clip(joint_vel, self.vel_lower_lim, self.vel_upper_lim)

        # Update the joint angles based on the velocity
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        new_joint_values = np.clip(new_joint_values, self.joint_lower_lim, self.joint_upper_lim)
        
        return new_joint_values



    def calc_jacobian(self, joint_values: list):
        """
        Calculates the Jacobian matrix for the robot based on the current joint values.

        Args:
            joint_values (list): The current joint values of the robot.

        Returns:
            jacobian (numpy.ndarray): The Jacobian matrix of the robot; linear velocity component only
        """

        l1, l2, l3, l4, l5, l6, l7 = self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7
        th1, th2, th3, th4, th5, th6 = joint_values if radians else np.rad2deg(joint_values)
        # order or variables: theta, d, a, alpha
        DH = np.array([[0, 0, 0, pi],
                       [th1, -l1-l2, 0, pi/2],
                       [-pi/2 + th2, 0, l3, pi],
                       [-pi/2 + th3, 0, 0, pi/2],
                       [th4, -l4-l5, 0, -pi/2],
                       [th5, 0, 0, pi/2],
                       [th6, -l6-l7, 0, pi]])
        
        H_B0, H_01, H_12, H_23, H_34, H_45, H_56 = [ut.dh_to_matrix(DH[i]) for i in range(len(joint_values)+1)]
        H_EE = H_B0@H_01@H_12@H_23@H_34@H_45@H_56  # Final transformation matrix for EE

        H_B1 = H_B0@H_01
        H_B2 = H_B1@H_12
        H_B3 = H_B2@H_23
        H_B4 = H_B3@H_34
        H_B5 = H_B4@H_45

        d_EE = H_EE[0:3, 3]
        k = np.array([0, 0, 1])
        jacobian = np.zeros(shape=(3, len(joint_values)))
        
        for i, H in enumerate([H_B0, H_B1, H_B2, H_B3, H_B4, H_B5]):
            z = H[0:3, 0:3]@k
            r = d_EE - H[0:3, 3]
            Jv = np.cross(z, r)
            jacobian[:, i] = Jv.T

        return jacobian

    def calc_inv_jacobian(self, joint_values: list):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        return np.linalg.pinv(self.calc_jacobian(joint_values))
    
    def calc_ik_single_soln(
        self, ee: ut.EndEffector, joint_values: List[float], soln: int = 0
    ):
        """
        Calculate the inverse kinematics for the HiWonder arm.
        NOTE: There will be 8 solutions for the arm

        Args:
            ee (EndEffector): Desired end effector position and orientation.
            joint_values (list): Current joint angles in radians.
            soln (int): Solution index for multiple IK solutions (if applicable).
        """
        base_config = (soln // 4) % 2 # 00001111
        elbow_config = (soln // 2) % 2 # 00110011
        wrist_config = soln % 2 # 01010101

        # Step 1: Compute wrist position
        l1, l2, l3, l4, l5, l6, l7 = self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7
        p_ee = np.array([ee.x, ee.y, ee.z])
        d6 = l6 + l7
        R_B6 = ut.euler_to_rotm((ee.rotx, ee.roty, ee.rotz))
        p_wrist = p_ee - d6*R_B6@np.array([0,0,1])
        wx,wy,wz = p_wrist

        # Step 2: Compute theta 1-3

        ## theta 1
        if base_config == 0:
            # Note the negative sign in front of atan2 is to account for the flipped z axis on joint 1
            th1 = -1*atan2(wy, wx)
            r = sqrt(wx**2 + wy**2)
        else:
            th1 = -1*atan2(-wy, -wx)
            r = -sqrt(wx**2 + wy**2)

        d1 = l1 + l2
        S = wz - d1
        L = sqrt(r**2 + S**2)
        try:
            beta_cos = (l3**2 + (l4+l5)**2 - L**2)/(2*l3*(l4+l5))
            beta = acos(beta_cos)
        except ValueError:
            print("Target is out of reach for the arm.")
            return joint_values

        ## theta 3
        if elbow_config == 0:
            th3 = pi - beta
        else:
            th3 = beta - pi
        
        ## theta 2
        psi = atan2(S,r)
        alpha = atan2((l4+l5)*sin(th3), l3+(l4+l5)*cos(th3))
        th2 = ut.wraptopi(pi/2 - psi + alpha)

        # Step 3: Compute R_B3
        R_B0 = ut.dh_to_matrix([0, 0, 0, pi])[0:3, 0:3]
        R_01 = ut.dh_to_matrix([th1, -l1-l2, 0, pi/2])[0:3, 0:3]
        R_12 = ut.dh_to_matrix([-pi/2 + th2, 0, l3, pi])[0:3, 0:3]
        R_23 = ut.dh_to_matrix([-pi/2 + th3, 0, 0, pi/2])[0:3, 0:3]
        R_B3 = R_B0@R_01@R_12@R_23

        R_23 = ut.dh_to_matrix([-pi/2 + th3, 0, 0, pi/2])[0:3, 0:3]
        # Step 4: Compute R_36
        R_36 = R_B3.T @ R_B6

        # Step 4: Compute theta 4-6

        ## Theta 4
        th4 = atan2(R_36[1,2], R_36[0,2])

        ## Theta 5
        c5 = -1*R_36[2,2]
        if wrist_config == 0:
            s5 = sqrt(1-c5**2)   
        else:
            s5 = -1 * sqrt(1-c5**2)

        th5 = atan2(s5,c5)

        ## Theta 6
        th6 = atan2(R_36[2,1], R_36[2,0])

        return [th1, th2, th3, th4, th5, th6]
    
    def calc_inverse_kinematics(self, ee: ut.EndEffector, joint_values: ut.List[float], soln: int = 0):
        soln_idx = [soln, (soln+1)%8, (soln+2)%8, (soln+3)%8, (soln+4)%8, (soln+5)%8, (soln+6)%8, (soln+7)%8]
        for i in soln_idx:
            soln_candidate = self.calc_ik_single_soln(ee, joint_values, soln = i)
            if ut.check_valid_ik_soln(soln_candidate, ee, self):
                return soln_candidate
        print("No valid IK solution found for the given end effector pose.")
        return joint_values

if __name__ == "__main__":
    
    model = Kinova6DOF()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
