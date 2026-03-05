from math import *
import traceback
import time
import numpy as np

from funrobo_hiwonder.core.hiwonder import HiwonderRobot #type: ignore

import funrobo_kinematics.core.utils as ut
from hiwonder_model import HiWonder5DOF

SQUARE_POSE = [
                np.array([60, 230, 0]),
                np.array([-60, 230, 0]),
                np.array([-60, 110, 0]),
                np.array([60, 110, 0])
            ]

SQUARE_POSE = [pose/1000 for pose in SQUARE_POSE]

STAR_POSE = [
                np.array([0, 230, 0]),
                np.array([38, 110, 0]),
                np.array([-60, 183, 0]),
                np.array([60, 183, 0]),
                np.array([-38, 110, 0])
]

STAR_POSE = [pose/1000 for pose in STAR_POSE]

Ry_0T = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [-1, 0, 0]])

Rz_0T = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

R_0T = Ry_0T@Rz_0T
d_0T = np.array([[-0.24], [0], [0]])

SQUARE_POSE_0T = [pose @ R_0T.T + d_0T.T for pose in SQUARE_POSE]
STAR_POSE_0T = [pose @ R_0T.T + d_0T.T for pose in STAR_POSE]

print("Precomputing IK for predefined poses...")
model = HiWonder5DOF()
SQUARE_JOINT_ANGLES = []
initial_guess = [0, 0, 0, 0, 0]
p_ee = ut.EndEffector()
pose = SQUARE_POSE_0T[0]
p_ee.x = pose[0][0]
p_ee.y = pose[0][1]
p_ee.z = pose[0][2]
SQUARE_JOINT_ANGLES.append(model.calc_numerical_ik(p_ee, initial_guess, tol=0.002, ilimit=1000))
for pose in SQUARE_POSE_0T[1:]:
    p_ee = ut.EndEffector()
    p_ee.x = pose[0][0]
    p_ee.y = pose[0][1]
    p_ee.z = pose[0][2]
    SQUARE_JOINT_ANGLES.append(model.calc_numerical_ik(p_ee, SQUARE_JOINT_ANGLES[-1], tol=0.002, ilimit=1000))

STAR_JOINT_ANGLES = []
for pose in STAR_POSE_0T:
    p_ee = ut.EndEffector()
    p_ee.x = pose[0][0]
    p_ee.y = pose[0][1]
    p_ee.z = pose[0][2]
    STAR_JOINT_ANGLES.append(model.calc_numerical_ik(p_ee, [0, 0, 0, 0, 0], tol=0.002, ilimit=1000))

print("Precomputing IK for predefined poses... Done.")



def run_ik():
    """ Main loop that reads gamepad commands and updates the robot accordingly. """
    try:

        # Initialize components
        robot = HiwonderRobot()
        model = HiWonder5DOF()
        
        control_hz = 20 
        dt = 1 / control_hz

        print("Waiting for initial joint reading...")

        pose_idx = 0
        while True:
            t_start = time.time()

            if robot.read_error is not None:
                print("[FATAL] Reader failed:", robot.read_error)
                break

            if robot.gamepad.cmdlist:
                cmd = robot.gamepad.cmdlist[-1]
                
                # Use home button to iterate through predefined poses
                
                if cmd.arm_home:
                    time.sleep(0.2)

                    # new_joint_values = model.calc_numerical_ik(p_ee, curr_joint_values, tol = 0.01, ilimit = 1000)
                    # print("new joint values (radians):", new_joint_values)

                    # new_joints_values = np.rad2deg(STAR_JOINT_ANGLES[pose_idx])
                    new_joints_values = np.rad2deg(SQUARE_JOINT_ANGLES[pose_idx % len(SQUARE_JOINT_ANGLES)])
                    # set new joint angles
                    # Need to add 6th joint position, though not controlled via rrmc. Use same as current value
                    all_joints_deg = np.append(new_joints_values, [0.0])

                    robot.set_joint_values(all_joints_deg, duration=dt, radians=False)
                    pose_idx += 1

            elapsed = time.time() - t_start
            remaining_time = dt - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)

            
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Initiating shutdown...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        robot.shutdown_robot()

if __name__ == "__main__":
    run_ik()
