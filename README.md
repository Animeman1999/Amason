Prototype a simulated autonomous mobile robot capable of traveling from a starting location to a destination while avoiding obstacles. 

The robot, named A.M.A.S.O.N. (Autonomous Mobile AI System for Optimized Navigation), was created in the Webots robotics simulator and programmed using Python. The robot uses a differential-drive design with two powered wheels and a rear caster. This should allow it to move forward, turn gradually, or rotate in place by changing the speed and direction of its left and right wheel motors. The three point design of two wheels and a caster also gives it stability. The simulated environment represents a small warehouse area containing a couple obstacles that the robot must navigate around.

## Current Itteration:
Basic robot is built.
Navigation via A* and reactive obstacle avoidance.

## Known Issues:
A* and reactive obstacle avoidance can get stuck in infinite turning loop.
