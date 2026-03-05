from math import *
import traceback
import time
import numpy as np

from funrobo_hiwonder.core.hiwonder import HiwonderRobot #type: ignore

import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.hiwonder5dof import HiWonder5DOF

def joystick_control(robot, dt, joint_values):
    """
    Apply joystick commands to control both the arm and the mobile base.

    This function reads the most recent gamepad command and performs two tasks: arm control
    and base control

    Args:
        robot: Instance of HiwonderRobot (RobotV5 or RobotV36).
        dt (float): Control timestep in seconds.
        joint_values (list[float]): Length-6 list of joint angle targets (degrees),
            maintained across loop iterations.

    Returns:
        list[float]: Updated joint_targets (degrees), to be fed back into the next call.

    Relevant notes:
        - `cmd.arm_j1 ... cmd.arm_ee` are assumed to be in [-1, 1].
    """

    cmd = robot.gamepad.cmdlist[-1]

    # ----------------------------------------------------------------------
    # Arm joint control
    # ----------------------------------------------------------------------

    max_rate = 400  # 400 x 0.1 = 40 deg/s
    joint_values[5] += dt * max_rate * cmd.arm_ee

    new_joint_values = robot.enforce_joint_limits(joint_values)
    new_joint_values = [round(theta,3) for theta in new_joint_values]

    print(f'[DEBUG] Commanded joint angles: [j1, j2, j3, j4, j5, ee]: {new_joint_values}')
    print(f'-------------------------------------------------------------------------------------\n')    
    
    # set new joint angles
    robot.set_joint_values(new_joint_values, duration=dt, radians=False)

    # ----------------------------------------------------------------------
    # base velocity control
    # ----------------------------------------------------------------------

    """
    Omni/mecanum-style wheel mixing (typical form):

        w0 = (vx - vy - w*L) / R
        w1 = (vx + vy + w*L) / R
        w2 = (vx + vy - w*L) / R
        w3 = (vx - vy + w*L) / R

    Where:
        - vx: desired forward velocity (m/s)
        - vy: desired lateral velocity (m/s)
        - w: desired yaw rate (rad/s)
        - R: wheel radius (m)
        - L: effective half-length + half-width (m), here approximated as
             (base_length_x + base_length_y)
    """
    vx, vy, w = cmd.base_vx, cmd.base_vy, cmd.base_w

    # Compute wheel speeds
    w0 = (vx - vy - w * (robot.base_length_x + robot.base_length_y)) / robot.wheel_radius
    w1 = (vx + vy + w * (robot.base_length_x + robot.base_length_y)) / robot.wheel_radius
    w2 = (vx + vy - w * (robot.base_length_x + robot.base_length_y)) / robot.wheel_radius
    w3 = (vx - vy + w * (robot.base_length_x + robot.base_length_y)) / robot.wheel_radius

    robot.set_wheel_speeds([w0, w1, w2, w3])


def run_rrmc():
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

                if cmd.arm_home:
                    print("Moving arm to home position")
                    robot.move_to_home_position()
                    continue 

                curr_joints_deg = robot.get_joint_values()

                if cmd.arm_ee != 0:
                    print("EE input")
                    joystick_control(robot, dt, curr_joints_deg)
                    continue


                curr_joints_rad = np.deg2rad(curr_joints_deg[:5])           
                vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]
                
                if all(abs(v) < 0.01 for v in vel):
                    # Zero velocity command; maintain current target
                    pass
                else:
                    # Integrate velocity to update target position
                    curr_joints_rad = model.calc_velocity_kinematics(curr_joints_rad, vel, dt=dt)

                new_joints_rad = curr_joints_rad
                new_joints_deg = np.rad2deg(new_joints_rad)
                print(new_joints_deg)

                # Need to add 6th joint position, though not controlled via rrmc. Use same as current value
                all_joints_deg = np.append(new_joints_deg, curr_joints_deg[5])

                robot.set_joint_values(all_joints_deg, duration=dt, radians=False)

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
    run_rrmc()
