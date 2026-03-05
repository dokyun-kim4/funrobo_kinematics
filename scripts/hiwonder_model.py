from math import *
import numpy as np
import random
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate

class HiWonder5DOF(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
        self.vel_lower_lim = np.array([l[0] for l in self.joint_vel_limits])
        self.vel_upper_lim = np.array([l[1] for l in self.joint_vel_limits])
        self.joint_lower_lim = np.array([l[0] for l in self.joint_limits])
        self.joint_upper_lim = np.array([l[1] for l in self.joint_limits])
    
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate the forward kinematics for the HiWonder arm.

        Args:
            joint_values (list): List of joint angles in radians or degrees.
            radians (bool): Whether the input joint angles are in radians.

        Returns:
            tuple: (EndEffector, list of H matrices for each joint)
        """
        curr_joint_values = joint_values.copy()
        DH = self.calc_dh(joint_values)

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

    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt: float = 0.02):
        """
        Calculate the new joint angles based on the desired end effector velocity.

        Args:
            joint_values (list): Current joint angles in radians.
            vel (list): Desired end effector velocity [vx, vy, vz].
            dt (float): Time step for the velocity update.
        
        Returns:
            new_joint_values (list): Updated joint angles in radians.
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


    def calc_jacobian(self, joint_values: list, radians=True):
        """
        Calculate the Jacobian of linear velocity for the HiWonder arm.

        Args:
            joint_values (list): Current joint angles in radians.
            radians (bool): Whether the input joint angles are in radians.

        Returns:
            np.ndarray: The Jacobian matrix.
        """

        DH = self.calc_dh(joint_values, radians=radians)
        
        H_LIST = [ut.dh_to_matrix(DH[i]) for i in range(len(joint_values))]
        H_01, H_12, H_23, H_34, H_45 = H_LIST
        H_EE = H_01@H_12@H_23@H_34@H_45  # Final transformation matrix for EE

        H_02 = H_01@H_12
        H_03 = H_02@H_23
        H_04 = H_03@H_34

        d_EE = H_EE[0:3, 3]
        k = np.array([0, 0, 1])
        jacobian = np.zeros(shape=(3, len(joint_values)))

        for i, H in enumerate([None, H_01, H_02, H_03, H_04]):
            if H is None:
                Jv = np.cross(k, d_EE)
            else:
                z = H[0:3, 0:3]@k
                r = d_EE - H[0:3, 3]
                Jv = np.cross(z, r)
            jacobian[:, i] = Jv.T

        return jacobian

    def calc_inv_jacobian(self, joint_values: list, lambda_: float = 0.05):
        """
        Calculate the pseudo-inverse of the Jacobian with dampening.

        Using the formula: J^T * (J * J^T + lambda^2 * I)^(-1)
        where lambda is the dampening factor to avoid singularities.

        Args:
            joint_values (list): Current joint angles in radians.
            lambda_ (float): Dampening factor
        
        Returns:
            np.ndarray: The pseudo-inverse of the Jacobian matrix.

        """
        J = self.calc_jacobian(joint_values)
        return J.T@ np.linalg.inv(J@J.T + (lambda_**2)*np.eye(J.shape[0]))

    def calc_dh(self, joint_values: list, radians=True):
        """
        Calculate the DH parameters for the HiWonder arm.

        Args:
            joint_values (list): List of joint angles in radians or degrees.
            radians (bool): Whether the input joint angles are in radians.
        
        Returns:
            np.ndarray: DH table
        """
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        th1,th2,th3,th4,th5 = joint_values if radians else np.rad2deg(joint_values)
        # order or variables: theta, d, a, alpha
        DH = np.array([[th1, l1, 0, -pi/2],
                       [th2-pi/2, 0, l2, pi],
                       [th3, 0, l3, pi],
                       [pi/2+th4, 0, 0, pi/2],
                       [th5, l4+l5, 0, 0]])
        return DH

    def calc_inverse_kinematics(self, ee: ut.EndEffector, joint_values: ut.List[float], soln: int = 0):
        """
        Calculate the inverse kinematics for the HiWonder arm.
        NOTE: There will be 4 solutions for the arm

        Args:
            ee (EndEffector): Desired end effector position and orientation.
            joint_values (list): Current joint angles in radians.
            soln (int): Solution index for multiple IK solutions (if applicable).
        """
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        th1_config = soln // 2
        elbow_config = soln % 2

        # Step 1: Compute wrist position
        R_05 = ut.euler_to_rotm((ee.rotx, ee.roty, ee.rotz))
        d5 = l4 + l5
        p_ee = np.array([ee.x, ee.y, ee.z])
        p_wrist = p_ee - d5*R_05@np.array([0,0,1])
        wx,wy,wz = p_wrist

        # Step 2: Compute theta 1-3

        ## Theta 1
        if th1_config == 0:
            th1 = atan2(wy, wx)
            r = sqrt(wx**2 + wy**2)
        else:
            th1 = atan2(-wy, -wx)
            r = -sqrt(wx**2 + wy**2)

        ## Theta 3
        S = wz - l1
        L = sqrt(r**2 + S**2)
        try:
            beta_cos = (l2**2 + l3**2 - L**2)/(2*l2*l3)
            print(beta_cos)
            beta = acos(beta_cos)
        except ValueError:
            print("Target is out of reach for the arm.")
            return joint_values

        if elbow_config == 0:
            th3 = pi - beta
        else:
            th3 = beta - pi

        ## Theta 2
        psi = atan2(S, r)
        alpha = atan2(l3*sin(th3), l2+l3*cos(th3))
        th2 = ut.wraptopi(pi/2 - psi + alpha)
        
        # Step 3: Compute R_03
        R_01 = ut.dh_to_matrix([th1, l1, 0, -pi/2])[0:3, 0:3]
        R_12 = ut.dh_to_matrix([th2-pi/2, 0, l2, pi])[0:3, 0:3]
        R_23 = ut.dh_to_matrix([th3, 0, l3, pi])[0:3, 0:3]
        R_03 = R_01 @ R_12 @ R_23

        # Step 4: Compute R_35
        R_35 = R_03.T @ R_05

        # Step 5: Compute theta 4-5

        ## Theta 4
        sin_th4 = R_35[1,2]
        cos_th4 = R_35[0,2]
        th4 = atan2(sin_th4, cos_th4)

        ## Theta 5
        sin_th5 = R_35[2,0]
        cos_th5 = R_35[2,1]
        th5 = atan2(sin_th5, cos_th5)
        
        return [th1, th2, th3, th4, th5]

    def calc_numerical_ik(self, ee: ut.EndEffector, joint_values: ut.List[float], tol: float = 0.002, ilimit: int = 1000):

        # Arm is underactuated, only do position IK
        p_ee = np.array([ee.x, ee.y, ee.z])
        # Get initial guess of joint values; ensure they are sampled from valid joint values.
        lim = [
                [-2*np.pi / 3, 2*np.pi / 3],
                [-2*np.pi / 3, 2*np.pi / 3],
                [-2*np.pi / 3, 2*np.pi / 3],
                [-2*np.pi / 3, 2*np.pi / 3],
                [-2*np.pi / 3, 2*np.pi / 3]
              ]
        
        # while True:
        guess = [
                random.uniform(*lim[0]), 
                random.uniform(*lim[1]),
                random.uniform(*lim[2]),
                random.uniform(*lim[3]),
                random.uniform(*lim[4]),
                ]
        icount = 0
        while icount < ilimit:
            fk_result, _ = self.calc_forward_kinematics(guess, True)
            diff = p_ee - np.array([fk_result.x, fk_result.y, fk_result.z])
            if np.linalg.norm(diff) < tol and ut.check_joint_limits(guess, lim):
                return guess
            guess += self.calc_inv_jacobian(guess)@diff
            icount += 1

if __name__ == "__main__":
    
    model = HiWonder5DOF()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
