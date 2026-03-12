# Mini Project 2
**Dexter Friis-Hecht and Dokyun Kim**

## Project Overview
In this project, we derived both the analytical and numerical inverse kinematics for two robot manipulator platforms: HiWonder's 5DOF arm and Kinova's 6DOF arm. The analytical and numerical implementations were tested in simulation for both platforms. For the HiWonder arm, they were also tested on real hardware.

## Intial Setup
Ensure the environment is set up correctly following the instructions in the [main branch](https://github.com/dokyun-kim4/funrobo_kinematics/tree/main?tab=readme-ov-file#step-by-step-setup). 

## How to run

### Part 1. Running the IK simulation
```bash
# For HiWonder
$ python scripts/hiwonder.py

# For Kinova
$ python scripts/kinova.py
```
The desired EE position can be set using the sliders on the left panel and pressing the "set pose" button. You can then run the analytical and numerical IK solvers by pressing the respective buttons.
Video below shows a sample demonstration  

[![HiWonder IK Demo Video](http://img.youtube.com/vi/ygeuNtyZRj8/0.jpg)](http://www.youtube.com/watch?v=ygeuNtyZRj8 "HiWoder Ik Demo")

[![Kinova IK Demo Video](http://img.youtube.com/vi/N38wuKvfXoc/0.jpg)](http://www.youtube.com/watch?v=N38wuKvfXoc "Kinova IK Demo")

### Part 2. Running the IK on real hardware
```bash
# SSH into the robot
$ ssh pi@<hostname>.local

# activate conda environment
$ cd funrobo_ws
$ conda activate funrobo_hw

# cd into the project directory to run script
$ cd funrobo_kinematics
$ python scripts/hiwonder_ik.py --shape <square/star/word>
```
Press the home button on the controller to move the robot to its next position. The video below shows a sample demonstration of the IK solvers running to trace shapes and words. 

[![HiWonder IK Demo Video](http://img.youtube.com/vi/isOjIMFhdZ0/0.jpg)](http://www.youtube.com/watch?v=isOjIMFhdZ0 "HiWoder Ik Demo on Hardware")
