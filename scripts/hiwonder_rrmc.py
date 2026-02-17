from math import *
import traceback
import time
import numpy as np

from funrobo_hiwonder.core.hiwonder import HiwonderRobot #type: ignore

import funrobo_kinematics.core.utils as ut
from hiwonder import HiWonder5DOF

def run_rrmc():
    """ Main loop that reads gamepad commands and updates the robot accordingly. """
    try:

        # Initialize components
        robot = HiwonderRobot()
        model = HiWonder5DOF()
        
        control_hz = 20 
        dt = 1 / control_hz

        print("Waiting for initial joint reading...")
        time.sleep(1)
        curr_joints_deg = robot.get_joint_values()
        if all(j == 0 for j in curr_joints_deg):
             print("[WARNING] Initial joint reading is all zeros. Check connection.")
        
        target_joints_rad = [np.deg2rad(j) for j in curr_joints_deg[:5]]
        print(f"Initial target joints (rad): {target_joints_rad}")

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
                vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]
                
                if all(abs(v) < 0.01 for v in vel):
                    # Zero velocity command; maintain current target
                    pass
                else:
                    # Integrate velocity to update target position
                    target_joints_rad = model.calc_velocity_kinematics(target_joints_rad, vel, dt=dt)

                new_joints_rad = target_joints_rad
                new_joints_deg = np.rad2deg(new_joints_rad)

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
