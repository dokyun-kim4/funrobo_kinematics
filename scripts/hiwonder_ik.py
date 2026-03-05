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
                np.array([-60, -110, 0]),
                np.array([60, -110, 0])
            ]

STAR_POSE = [
                np.array([0, 230, 0]),
                np.array([38, -110, 0]),
                np.array([-60, 183, 0]),
                np.array([60, 183, 0]),
                np.array([-38, -110, 0])
]

Ry_0T = np.array([[0, 0, -1],
                  [0, 1, 0],
                  [1, 0, 0]])

Rz_0T = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 1]])

R_0T = Ry_0T@Rz_0T
d_0T = np.array([[0.24], [0], [0]])

SQUARE_POSE_0T = [pose @ R_0T.T + d_0T.T for pose in SQUARE_POSE]
STAR_POSE_0T = [pose @ R_0T.T + d_0T.T for pose in STAR_POSE]


def run_ik():
    """ Main loop that reads gamepad commands and updates the robot accordingly. """
    try:

        # Initialize components
        robot = HiwonderRobot()
        model = HiWonder5DOF()
        
        control_hz = 20 
        dt = 1 / control_hz

        print("Waiting for initial joint reading...")

        while True:
            t_start = time.time()

            if robot.read_error is not None:
                print("[FATAL] Reader failed:", robot.read_error)
                break

            if robot.gamepad.cmdlist:
                cmd = robot.gamepad.cmdlist[-1]
                
                # Use home button to iterate through predefined poses
                pose_idx = 0
                if cmd.arm_home:
                    print("next pose")
                    p_ee = ut.EndEffector()
                    curr_goal_pos = SQUARE_POSE_0T[pose_idx]
                    print(curr_goal_pos)
                    curr_joint_values = np.deg2rad(robot.get_joint_values())
                    p_ee.x = curr_goal_pos[0][0]
                    p_ee.y = curr_goal_pos[0][1]
                    p_ee.z = curr_goal_pos[0][2]
                    new_joint_values = model.calc_numerical_ik(p_ee, curr_joint_values, tol = 0.01, ilimit = 1000)
                    print("new joint values (radians):", new_joint_values)

                    new_joints_values = np.rad2deg(new_joint_values)
                    # set new joint angles
                    robot.set_joint_values(new_joint_values, duration=dt, radians=False)

                    # Need to add 6th joint position, though not controlled via rrmc. Use same as current value
                    all_joints_deg = np.append(new_joints_deg, curr_joints_deg[5])

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
