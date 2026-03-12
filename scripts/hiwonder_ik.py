from math import *
import traceback
import time
import numpy as np
import copy
import argparse

from funrobo_hiwonder.core.hiwonder import HiwonderRobot #type: ignore

import funrobo_kinematics.core.utils as ut
from hiwonder_model import HiWonder5DOF

def run_ik(joint_angles=None):
    """ Main loop that reads gamepad commands and updates the robot accordingly. """
    try:

        # Initialize components
        robot = HiwonderRobot()
        
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
                    time.sleep(0.4) # prevents multiple presses from registering
                    
                    if joint_angles is not None:
                        new_joints_values = np.rad2deg(joint_angles[pose_idx % len(joint_angles)])
                        # Need to add 6th joint position for gripper
                        all_joints_deg = np.append(new_joints_values, [0.0])

                        robot.set_joint_values(all_joints_deg, duration=0.5, radians=False)

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

    parser = argparse.ArgumentParser(description='Run Hiwonder IK with specific shape')
    parser.add_argument('--shape', type=str, default='square', choices=['square', 'star', 'word'], help='Shape to trace: square, star, or word')
    args = parser.parse_args()
    
    # Pre-compute IK solutions for the predefined poses to speed up execution during runtime
    model = HiWonder5DOF()

    SQUARE_POSE = [
                    np.array([0.060, 0.260, 0]),
                    np.array([-0.060, 0.260, 0]),
                    np.array([-0.060, 0.140, 0]),
                    np.array([0.060, 0.140, 0])
                ]

    SQUARE_EE = []

    STAR_POSE = [
                    np.array([0, 0.270, 0]),
                    np.array([0.038, 0.150, 0]),
                    np.array([-0.060, 0.223, 0]),
                    np.array([0.060, 0.223, 0]),
                    np.array([-0.038, 0.150, 0])
                ]

    WORD_POSE = [
                    # Letter C
                    np.array([-0.07, 0.18, 0]),
                    np.array([-0.12, 0.18, 0]),
                    np.array([-0.12, 0.09, 0]),
                    np.array([-0.07, 0.09, 0]),

                    # Letter U
                    np.array([-0.03, 0.18, 0]),
                    np.array([-0.03, 0.09, 0]),
                    np.array([0.03, 0.09, 0]),
                    np.array([0.03, 0.18, 0]),

                    # Letter T
                    np.array([0.06, 0.18, 0 ]),
                    np.array([0.12, 0.18, 0]),
                    np.array([0.09, 0.18, 0]),
                    np.array([0.09, 0.09, 0])
                ]

    # Transformation matrices from robot base to target frame
    Ry_0T = np.array([[0, 0, 1],
                    [0, 1, 0],
                    [-1, 0, 0]])

    Rz_0T = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

    R_0T = Ry_0T@Rz_0T
    d_0T = np.array([[-0.26], [0], [0]])

    SQUARE_POSE_0T = [pose @ R_0T.T + d_0T.T for pose in SQUARE_POSE]
    STAR_POSE_0T = [pose @ R_0T.T + d_0T.T for pose in STAR_POSE]
    WORD_POSE_0T = [pose @ R_0T.T + d_0T.T for pose in WORD_POSE]

    print("Precomputing IK for predefined poses...")
    initial_guess = [0.0, 0.0, pi/2, -pi/6, 0.0]
    SQUARE_JOINT_ANGLES = []
    p_ee = ut.EndEffector()
    p_ee.x, p_ee.y, p_ee.z = SQUARE_POSE_0T[0][0]

    soln_sq = model.calc_numerical_ik(p_ee, initial_guess, tol=0.002, ilimit=1000)
    SQUARE_JOINT_ANGLES.append(copy.deepcopy(soln_sq))
    for pose in SQUARE_POSE_0T[1:]:
        p_ee = ut.EndEffector()
        p_ee.x, p_ee.y, p_ee.z = pose[0]
        soln_sq = model.calc_numerical_ik(p_ee, copy.deepcopy(SQUARE_JOINT_ANGLES[-1]), tol=0.002, ilimit=1000)
        SQUARE_JOINT_ANGLES.append(copy.deepcopy(soln_sq))

    STAR_JOINT_ANGLES = []
    p_ee = ut.EndEffector()
    p_ee.x, p_ee.y, p_ee.z = STAR_POSE_0T[0][0]
    soln_st = model.calc_numerical_ik(p_ee, initial_guess, tol=0.002, ilimit=1000)
    STAR_JOINT_ANGLES.append(copy.deepcopy(soln_st))
    for pose in STAR_POSE_0T[1:]:
        p_ee = ut.EndEffector()
        p_ee.x, p_ee.y, p_ee.z = pose[0]
        soln_st = model.calc_numerical_ik(p_ee, copy.deepcopy(STAR_JOINT_ANGLES[-1]), tol=0.002, ilimit=1000)
        STAR_JOINT_ANGLES.append(copy.deepcopy(soln_st))

    WORD_JOINT_ANGLES = []
    p_ee = ut.EndEffector()
    p_ee.x, p_ee.y, p_ee.z = WORD_POSE_0T[0][0]
    soln = model.calc_numerical_ik(p_ee, initial_guess, tol=0.002, ilimit=1000)

    WORD_JOINT_ANGLES.append(copy.deepcopy(soln))
    for pose in WORD_POSE_0T[1:]:
        p_ee = ut.EndEffector()
        p_ee.x, p_ee.y, p_ee.z = pose[0]
        soln = model.calc_numerical_ik(p_ee, copy.deepcopy(WORD_JOINT_ANGLES[-1]), tol=0.002, ilimit=1000)
        WORD_JOINT_ANGLES.append(copy.deepcopy(soln))

    print("Precomputing IK for predefined poses... Done.")

    if args.shape == 'square':
        target_joints = SQUARE_JOINT_ANGLES
    elif args.shape == 'star':
        target_joints = STAR_JOINT_ANGLES
    elif args.shape == 'word':
        target_joints = WORD_JOINT_ANGLES
    else:
        target_joints = SQUARE_JOINT_ANGLES

    run_ik(target_joints)
