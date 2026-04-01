# Individual HW #2:  Trajectory Generation Methods
Dokyun Kim

## Assignment overview
For this assignment, I implemented two types of trajectory generation methods:
1. Quintic polynomial
2. Trapezoidal

## Quintic Polynomial
The implementation of the quintic polynomial generation method can be found [here](https://github.com/dokyun-kim4/funrobo_kinematics/blob/6ca45612e42a4d8671d23105357950112a13fb3f/examples/traj_gen.py#L91).

The image below shows the generated position, velocity, acceleration plots for both task-space and joint-space trajectories.

<image src="media/quintic_joint_space.png" alt="Quintic Trajectory; Joint Space" width="800"/>
<image src="media/quintic_task_space.png" alt="Quintic Trajectory; Task Space" width="800"/>

The video below demonstrates the simulated 5DOF arm following the two different trajectories.

[![Joint Space Demo Video](http://img.youtube.com/vi/lHZ7wSkBSTk/0.jpg)](https://www.youtube.com/watch?v=lHZ7wSkBSTk "Joint Space Demo")  

[![Task Space Demo Video](http://img.youtube.com/vi/_WDXWWWYIFY/0.jpg)](http://www.youtube.com/watch?v=_WDXWWWYIFY "Task Space Demo")

## Trapezoidal
The implementation of the trapezoidal generation method can be found [here](https://github.com/dokyun-kim4/funrobo_kinematics/blob/6ca45612e42a4d8671d23105357950112a13fb3f/examples/traj_gen.py#L182).

The image below shows the generated position, velocity, and acceleration plots for both task-space and joint-space trajectories

<image src="media/trapezoidal_joint_space.png" alt="Trapezoidal Trajectory; Joint Space" width="800"/>
<image src="media/trapezoidal_task_space.png" alt="Trapezoidal Trajectory; Task Space" width="800"/>

The video below demonstrates the simulated 5DOF arm following the two different trajectories.

[![Joint Space Demo Video](http://img.youtube.com/vi/E-VdqmnlT_k/0.jpg)](https://www.youtube.com/watch?v=E-VdqmnlT_k "Joint Space Demo")  

[![Task Space Demo Video](http://img.youtube.com/vi/WYOINWk4CjI/0.jpg)](http://www.youtube.com/watch?v=WYOINWk4CjI "Task Space Demo")
