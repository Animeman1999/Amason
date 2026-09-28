# A.M.A.S.O.N. Autonomous Navigation Project

The robot, named A.M.A.S.O.N. (Autonomous Mobile AI System for Optimized Navigation), was created in the Webots robotics simulator and programmed using Python. The robot uses a differential-drive design with two powered wheels and a rear caster. This should allow it to move forward, turn gradually, or rotate in place by changing the speed and direction of its left and right wheel motors. The three point design of two wheels and a caster also gives it stability. The simulated environment represents a small warehouse area containing a couple obstacles that the robot must navigate around.

## Current Iteration:
Basic robot is built.
Navigation via A* and reactive obstacle avoidance.Dynamic obstacle detection and A* replanning.
Finite-state-machine decision making.
Automated multi-point patrol behavior.
A* path simplification to reduce unnecessary waypoints.
CSV performance logging and route/sensor visualization.

## Known Issues:
None

## Running the Path Planning Code

### Requirements
- Webots R2025a or compatible version
- Python 3.12
- A Webots world containing:
  - Robot named `AMASON`
  - Left wheel motor named `left wheel motor`
  - Right wheel motor named `right wheel motor`
  - GPS named `gps`
  - InertialUnit named `inertial unit`
  - Distance sensors named:
    - `front sensor`
    - `left front sensor`
    - `right front sensor`
- Controller file named `amason_controller.py`

### Instructions
1. Open the A.M.A.S.O.N. world in Webots.
2. Make sure the robot controller is set to amason_controller.
3. Confirm that the robot starts at the position expected by the controller.
4. Confirm that the known static obstacles in the Webots world match the obstacle coordinates defined in the controller.
5. Reset the simulation to time 0.
6. Run the simulation.
7. Watch the Webots console for:
   - The initial A* path
   - Simplified navigation waypoints
   - Patrol point changes
   - Obstacle detections
   - Temporary obstacle mapping
   - A* replanning events
   - Finite-state-machine state changes
The controller divides the 5 m × 5 m arena into a 20 × 20 occupancy grid using 0.25 m cells. A* searches through open cells while avoiding blocked cells. The resulting path is simplified to remove unnecessary intermediate grid cells and then converted into world-coordinate waypoints that the robot follows using GPS position and inertial-unit heading.
The robot now operates as an autonomous patrol robot. It moves between a series of predefined patrol locations. When one patrol point is reached, the controller automatically selects the next patrol point and calculates a new A* route.
If the distance sensors detect an unexpected obstacle, the robot stops briefly to confirm the object is still present. The controller estimates the obstacle position using the robot's current position, heading, sensor angle, and measured distance. The obstacle is temporarily added to the occupancy grid, and A* recalculates a new path from the robot's current location to the current patrol point.
Previously detected temporary obstacles are remembered so that the same obstacle does not continuously trigger replanning. Temporary obstacles expire after a period of time if they are no longer detected.
The controller also creates a CSV performance log named amason_navigation_log.csv. A separate program named visualize_amason_log.py can be used to display the robot's traveled route and distance-sensor readings.

## Design Document Summary

### Section 1: Path Planning
The path-planning system uses the **A* search algorithm** with a **Manhattan-distance heuristic**. Known obstacles are represented in an occupancy grid with safety margins around them. The path is simplified to remove unnecessary intermediate waypoints before the robot follows it.

### Section 2: Probabilistic Localization
The localization design proposes **Monte Carlo Localization (Particle Filtering)**. Each particle represents a possible robot position and heading. A motion model introduces uncertainty caused by wheel slip and imperfect movement, while a sensor model compares predicted distance readings with actual readings. Particles with higher probabilities are retained through resampling, allowing the estimated position to converge over time.

### Section 3: Real-Time Obstacle Avoidance
The real-time obstacle-avoidance system combines distance-sensor detection with dynamic A* replanning. Unexpected obstacles are confirmed, mapped as temporary blocked cells, and added to the planning grid. A* then calculates a new route from the robot's current position. Previously detected temporary obstacles are remembered so that the same object does not continuously trigger replanning.

## Main AI Techniques
A* search for global path planning
Manhattan-distance heuristic
A* path simplification
Finite-state-machine navigation
Automated patrol task execution
Sensor-based obstacle detection
Dynamic occupancy-grid updates
Real-time A* replanning
Temporary obstacle memory
CSV performance logging
Route and sensor visualization
