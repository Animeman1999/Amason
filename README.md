# A.M.A.S.O.N. Autonomous Navigation Project

The robot, named A.M.A.S.O.N. (Autonomous Mobile AI System for Optimized Navigation), was created in the Webots robotics simulator and programmed using Python. The robot uses a differential-drive design with two powered wheels and a rear caster. This should allow it to move forward, turn gradually, or rotate in place by changing the speed and direction of its left and right wheel motors. The three point design of two wheels and a caster also gives it stability. The simulated environment represents a small warehouse area containing known static obstacles and temporary obstacles that the robot must navigate around.

## Current Iteration:
Basic robot is built.
Corner-to-corner navigation using A* path planning.
Patrolling remains in the controller but is currently turned off.
Noisy GPS and distance-sensor data are simulated.
Markov localization with Bayesian correction is used to estimate the robot's position.
A 20 × 20 occupancy grid represents the warehouse environment.
Known green obstacles are loaded into the map before navigation begins.
Temporary red obstacles are detected and added to the occupancy grid dynamically.
Dynamic obstacle detection and A* replanning are implemented.
Finite-state-machine decision making controls navigation and recovery behaviors.
A* path simplification reduces unnecessary waypoints.
Local obstacle avoidance uses backtracking, escape turning, and escape driving.
The robot can detect when it is physically stuck by monitoring forward progress.
CSV performance logging and route/sensor visualization are included.

## Known Issues:
None currently identified.

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
3. Confirm that the robot starts near the corner expected by the controller.
4. Confirm that the known static obstacles in the Webots world match the obstacle coordinates defined in the controller.
5. Temporary red obstacles should not be added to the controller's static obstacle map.
6. Reset the simulation to time 0.
7. Run the simulation.
8. Watch the Webots console for:
   - The initial A* path
   - Simplified navigation waypoints
   - Localization estimates and confidence values
   - Obstacle detections
   - Temporary obstacle mapping
   - A* replanning events
   - Backtrack and escape behaviors
   - Stuck detection and recovery
   - Finite-state-machine state changes

The controller divides the 5 m × 5 m arena into a 20 × 20 occupancy grid using 0.25 m cells. A* searches through open cells while avoiding blocked cells. The resulting path is simplified to remove unnecessary intermediate grid cells and then converted into world-coordinate waypoints.

The current test places the robot near one corner of the warehouse and gives it a single goal near the opposite corner. The automated patrol system remains in the controller but is currently disabled so that the robot can be tested on a more difficult corner-to-corner route.

The localization system no longer relies only on perfect Webots GPS data. Gaussian noise is added to the simulated GPS and distance-sensor readings. A grid-based Markov localization system predicts the robot's possible location based on movement and then uses a Bayesian correction based on the noisy GPS measurement. The controller calculates an estimated position, most likely grid cell, and localization confidence.

The mapping system uses the occupancy grid to represent the environment. Known green obstacles are added to the map before navigation begins. The robot also maintains information about cells it has observed while moving. The front, left-front, and right-front distance sensors are used to identify free space and detect unexpected obstacles.

If the distance sensors detect an unexpected red obstacle, the robot stops briefly to confirm that the object is still present. The controller estimates the obstacle position using the robot's estimated position, heading, sensor angle, and measured distance. The obstacle is temporarily added to the occupancy grid, and A* recalculates a new path from the robot's current location to the goal.

Previously detected temporary obstacles are remembered for 120 seconds so that the same obstacle does not repeatedly disappear from the map while the robot is still navigating around it. Nearby detections are matched so that noisy sensor readings are less likely to create duplicate temporary obstacles.

The obstacle-avoidance system also includes local recovery behavior. If the robot becomes too close to an obstacle, it can back up, turn toward the clearer side, drive away from the blocked area, and then calculate a new A* route. The recovery sequence is BACKTRACK, ESCAPE_TURN, ESCAPE_DRIVE, and PLAN_ROUTE.

The current controller can also recognize when the robot is physically stuck even if the front distance sensor does not clearly detect the problem. While driving toward a waypoint, the controller monitors whether the robot is making enough forward progress. If it continues attempting to move without getting significantly closer to the waypoint, it performs a longer backtrack and escape maneuver before replanning.

The controller also creates a CSV performance log named amason_navigation_log.csv. A separate program named visualize_amason_log.py can be used to display the robot's traveled route and distance-sensor readings.

## Design Document Summary

### Section 1: Path Planning
The path-planning system uses the **A* search algorithm** with a **Manhattan-distance heuristic**. Known obstacles, arena boundaries, and detected temporary obstacles are represented in an occupancy grid with safety margins around them. The path is simplified to remove unnecessary intermediate waypoints before the robot follows it. When the environment changes, A* recalculates a route from the robot's current estimated location to the goal.

### Section 2: Probabilistic Localization
The current localization system implements **Markov localization with Bayesian correction**. Simulated noise is added to GPS measurements so that the robot must estimate its position instead of receiving a perfect location. A motion model predicts how the robot's location probability changes as it moves. A Bayesian measurement update then compares the predicted position probabilities with the noisy GPS reading. The result is an estimated world position, a most likely grid cell, and a localization confidence value.

### Section 3: Mapping and Environment Understanding
The warehouse is represented by a **20 × 20 occupancy grid**. Known green obstacles are entered into the static map before the simulation begins. As the robot moves, sensor readings are used to observe free and occupied cells. Unexpected red obstacles are estimated in world coordinates, converted to grid cells, and stored as temporary obstacles. These temporary obstacles are then included in future A* path calculations.

### Section 4: Real-Time Obstacle Avoidance
The real-time obstacle-avoidance system combines distance-sensor detection with dynamic A* replanning and local reactive behavior. Unexpected obstacles are confirmed, mapped as temporary blocked cells, and added to the planning grid. A* then calculates a new route from the robot's current estimated position.

If the robot becomes too close to an obstacle or cannot safely continue, it can backtrack, turn toward the clearer side, drive away from the blocked location, and replan. The controller also includes a no-progress stuck detector. If the robot is commanded to move forward but does not make enough progress toward the waypoint, it automatically performs a recovery maneuver before trying another route.

## Main AI Techniques
A* search for global path planning
Manhattan-distance heuristic
A* path simplification
Markov localization
Bayesian position correction
Simulated noisy GPS and distance-sensor data
Finite-state-machine navigation
Occupancy-grid mapping
Sensor-based obstacle detection
Dynamic occupancy-grid updates
Real-time A* replanning
Temporary obstacle memory
Local backtracking and escape behavior
No-progress stuck detection and recovery
CSV performance logging
Route and sensor visualization
