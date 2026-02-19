# Mini Project 1
**Dexter Friis-hecht and Dokyun Kim**

## Project Overview
In this project, we applied the Denavit-Hartenberg (DH) convention to determine forward kinematics equations for two robot manipulator platforms: HiWonder's 5DOF arm and Kinova's 6DOF arm. The DH parameters were then used to derive Jacobian matrices to implement Resolved-Rate Motor Control (RRMC) for the HiWondder arm via simulation and real hardware.

## Intial Setup
Ensure the environment is set up correctly following the instructions in the [main branch](https://github.com/dokyun-kim4/funrobo_kinematics/tree/main?tab=readme-ov-file#step-by-step-setup). 

## How to run

### 1. Running the FPK simulation

```bash
# For HiWonder
$ python scripts/hiwonder.py

# For Kinova
$ python scripts/kinova.py
```
The robot can be controlled using the sliders on the left panel.  
Video below shows a sample demonstration  

[![FPK Demo Video](http://img.youtube.com/vi/YY2ypagi3Zw/0.jpg)](http://www.youtube.com/watch?v=YY2ypagi3Zw "FPK Demo")

### 2. Running the RRMC simulation
```bash
$ python scripts/hiwonder.py
```
For velocity control, activate VK using the button on the left panel. Then, control the velocity through keyboard input.  
Video below shows a sample demonstration  

[![FVK Demo Video](http://img.youtube.com/vi/gagwGi68yLU/0.jpg)](http://www.youtube.com/watch?v=gagwGi68yLU "FVK Demo")

### Running the RRMC on real hardware
```bash
# SSH into the robot
$ ssh pi@<hostname>.local

# activate conda environment
$ cd funrobo_ws
$ conda activate funrobo_hw

# cd into the project directory to run script
$ cd funrobo_kinematics
$ python scripts/hiwonder_rrmc.py
```

The robot can be controlled using the gamepad.  
Video below shows a sample demonstration  

[![FVK HW Demo Video](http://img.youtube.com/vi/UIM5e9WNGxA/0.jpg)](http://www.youtube.com/watch?v=UIM5e9WNGxA "FVK HW Demo")
