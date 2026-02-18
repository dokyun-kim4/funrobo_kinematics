from math import *
import numpy as np
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

        DH = self.calc_dh(joint_values)
        
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

    def calc_inv_jacobian(self, joint_values: list):
        lam = 0.05
        J = self.calc_jacobian(joint_values)
        return J.T@ np.linalg.inv(J@J.T+lam**2*np.eye(J.shape[0]))
        #return np.linalg.pinv(jacobian, rcond=1e-2)

    def calc_dh(self, joint_values: list):
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        th1,th2,th3,th4,th5 = joint_values if radians else np.rad2deg(joint_values)
        # order or variables: theta, d, a, alpha
        DH = np.array([[th1, l1, 0, -pi/2],
                       [th2-pi/2, 0, l2, pi],
                       [th3, 0, l3, pi],
                       [pi/2+th4, 0, 0, pi/2],
                       [th5, l4+l5, 0, 0]])
        return DH


if __name__ == "__main__":
    
    model = HiWonder5DOF()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
