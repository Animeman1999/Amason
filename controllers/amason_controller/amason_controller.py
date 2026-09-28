"""A.M.A.S.O.N. Webots Controller

Autonomous Mobile AI System for Optimized Navigation

Adds to the previous controller:
- autonomous patrol task execution
- simplified A* waypoint paths
- A* closed-set optimization
- dynamic obstacle memory and replanning
- reduced-rate console output
- CSV performance/data logging
"""

from controller import Robot
import csv
import heapq
import math
import os


# =========================================================
# ROBOT AND DEVICES
# =========================================================

amason_robot = Robot()
SIMULATION_TIME_STEP = int(amason_robot.getBasicTimeStep())

left_wheel_motor = amason_robot.getDevice("left wheel motor")
right_wheel_motor = amason_robot.getDevice("right wheel motor")
left_wheel_motor.setPosition(float("inf"))
right_wheel_motor.setPosition(float("inf"))
left_wheel_motor.setVelocity(0.0)
right_wheel_motor.setVelocity(0.0)

gps_sensor = amason_robot.getDevice("gps")
inertial_orientation_sensor = amason_robot.getDevice("inertial unit")
gps_sensor.enable(SIMULATION_TIME_STEP)
inertial_orientation_sensor.enable(SIMULATION_TIME_STEP)

front_distance_sensor = amason_robot.getDevice("front sensor")
left_front_distance_sensor = amason_robot.getDevice("left front sensor")
right_front_distance_sensor = amason_robot.getDevice("right front sensor")
front_distance_sensor.enable(SIMULATION_TIME_STEP)
left_front_distance_sensor.enable(SIMULATION_TIME_STEP)
right_front_distance_sensor.enable(SIMULATION_TIME_STEP)


# =========================================================
# MOVEMENT AND NAVIGATION SETTINGS
# =========================================================

FORWARD_WHEEL_SPEED = 3.0
TURNING_WHEEL_SPEED = 1.5
WAYPOINT_HEADING_TOLERANCE = math.radians(5)
WAYPOINT_DISTANCE_TOLERANCE = 0.10

OBSTACLE_DETECTION_THRESHOLD = 300
OBSTACLE_CLEAR_THRESHOLD = 350
OBSTACLE_CONFIRMATION_TIME_SECONDS = 0.50
TEMPORARY_OBSTACLE_LIFETIME_SECONDS = 30.0
TEMPORARY_OBSTACLE_INFLATION_CELLS = 1
TEMPORARY_OBSTACLE_MATCH_RADIUS_CELLS = 1
PATH_RETRY_INTERVAL_SECONDS = 1.0


# =========================================================
# SENSOR MOUNTING LOCATIONS
# =========================================================

FRONT_SENSOR_LOCAL_X = 0.255
FRONT_SENSOR_LOCAL_Y = 0.0
FRONT_SENSOR_ANGLE = 0.0

LEFT_SENSOR_LOCAL_X = 0.245
LEFT_SENSOR_LOCAL_Y = 0.15
LEFT_SENSOR_ANGLE = math.radians(30)

RIGHT_SENSOR_LOCAL_X = 0.245
RIGHT_SENSOR_LOCAL_Y = -0.15
RIGHT_SENSOR_ANGLE = math.radians(-30)


# =========================================================
# ARENA AND MAP SETTINGS
# =========================================================

ARENA_SIZE_METERS = 5.0
GRID_CELL_SIZE_METERS = 0.25
GRID_CELL_COUNT = int(ARENA_SIZE_METERS / GRID_CELL_SIZE_METERS)
ARENA_MINIMUM_COORDINATE = -ARENA_SIZE_METERS / 2.0

BOX_ONE_CENTER_X = 1.20
BOX_ONE_CENTER_Y = 0.40
BOX_ONE_SIZE_X = 0.20
BOX_ONE_SIZE_Y = 0.40

BOX_TWO_CENTER_X = 0.70
BOX_TWO_CENTER_Y = 1.10
BOX_TWO_SIZE_X = 0.20
BOX_TWO_SIZE_Y = 0.40

STATIC_OBSTACLE_SAFETY_MARGIN = 0.30

ROBOT_START_WORLD_POSITION = (0.0, 0.0)

# Autonomous patrol route. The robot cycles through these forever.
PATROL_LOCATIONS = [
    (1.80, 1.20),
    (-1.50, 1.50),
    (-1.50, -1.50),
    (1.50, -1.50),
]


# =========================================================
# LOGGING SETTINGS
# =========================================================

LOG_INTERVAL_SECONDS = 0.10
CONSOLE_STATUS_INTERVAL_SECONDS = 0.50
CONTROLLER_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(CONTROLLER_DIRECTORY, "amason_navigation_log.csv")


# =========================================================
# BASIC HELPER FUNCTIONS
# =========================================================

def normalize_angle_radians(angle_radians):
    while angle_radians > math.pi:
        angle_radians -= 2 * math.pi
    while angle_radians < -math.pi:
        angle_radians += 2 * math.pi
    return angle_radians


def convert_world_position_to_grid_cell(world_x_position, world_y_position):
    grid_column = int((world_x_position - ARENA_MINIMUM_COORDINATE) / GRID_CELL_SIZE_METERS)
    grid_row = int((world_y_position - ARENA_MINIMUM_COORDINATE) / GRID_CELL_SIZE_METERS)
    grid_column = max(0, min(GRID_CELL_COUNT - 1, grid_column))
    grid_row = max(0, min(GRID_CELL_COUNT - 1, grid_row))
    return grid_column, grid_row


def convert_grid_cell_to_world_position(grid_column, grid_row):
    world_x_position = ARENA_MINIMUM_COORDINATE + (grid_column + 0.5) * GRID_CELL_SIZE_METERS
    world_y_position = ARENA_MINIMUM_COORDINATE + (grid_row + 0.5) * GRID_CELL_SIZE_METERS
    return world_x_position, world_y_position


def create_empty_occupancy_grid():
    return [[0 for _ in range(GRID_CELL_COUNT)] for _ in range(GRID_CELL_COUNT)]


def copy_occupancy_grid(source_grid):
    return [grid_column.copy() for grid_column in source_grid]


# =========================================================
# OBSTACLE MAP FUNCTIONS
# =========================================================

def mark_rectangular_obstacle_on_grid(
    occupancy_grid,
    obstacle_center_x,
    obstacle_center_y,
    obstacle_size_x,
    obstacle_size_y,
    safety_margin,
):
    obstacle_minimum_x = obstacle_center_x - obstacle_size_x / 2.0 - safety_margin
    obstacle_maximum_x = obstacle_center_x + obstacle_size_x / 2.0 + safety_margin
    obstacle_minimum_y = obstacle_center_y - obstacle_size_y / 2.0 - safety_margin
    obstacle_maximum_y = obstacle_center_y + obstacle_size_y / 2.0 + safety_margin

    minimum_grid_column, minimum_grid_row = convert_world_position_to_grid_cell(
        obstacle_minimum_x, obstacle_minimum_y
    )
    maximum_grid_column, maximum_grid_row = convert_world_position_to_grid_cell(
        obstacle_maximum_x, obstacle_maximum_y
    )

    for obstacle_grid_column in range(minimum_grid_column, maximum_grid_column + 1):
        for obstacle_grid_row in range(minimum_grid_row, maximum_grid_row + 1):
            occupancy_grid[obstacle_grid_column][obstacle_grid_row] = 1


def mark_temporary_obstacle_on_grid(occupancy_grid, obstacle_grid_cell):
    obstacle_grid_column, obstacle_grid_row = obstacle_grid_cell

    for column_offset in range(
        -TEMPORARY_OBSTACLE_INFLATION_CELLS,
        TEMPORARY_OBSTACLE_INFLATION_CELLS + 1,
    ):
        for row_offset in range(
            -TEMPORARY_OBSTACLE_INFLATION_CELLS,
            TEMPORARY_OBSTACLE_INFLATION_CELLS + 1,
        ):
            blocked_grid_column = obstacle_grid_column + column_offset
            blocked_grid_row = obstacle_grid_row + row_offset

            if (
                0 <= blocked_grid_column < GRID_CELL_COUNT
                and 0 <= blocked_grid_row < GRID_CELL_COUNT
            ):
                occupancy_grid[blocked_grid_column][blocked_grid_row] = 1


temporary_obstacles = {}


def remove_expired_temporary_obstacles(current_simulation_time):
    expired_obstacle_cells = []

    for obstacle_grid_cell, last_detection_time in temporary_obstacles.items():
        obstacle_age_seconds = current_simulation_time - last_detection_time
        if obstacle_age_seconds > TEMPORARY_OBSTACLE_LIFETIME_SECONDS:
            expired_obstacle_cells.append(obstacle_grid_cell)

    for obstacle_grid_cell in expired_obstacle_cells:
        del temporary_obstacles[obstacle_grid_cell]
        print("Temporary obstacle expired:", obstacle_grid_cell)


def find_matching_temporary_obstacle(detected_grid_cell):
    detected_grid_column, detected_grid_row = detected_grid_cell

    for existing_grid_cell in temporary_obstacles.keys():
        column_difference = abs(existing_grid_cell[0] - detected_grid_column)
        row_difference = abs(existing_grid_cell[1] - detected_grid_row)

        if (
            column_difference <= TEMPORARY_OBSTACLE_MATCH_RADIUS_CELLS
            and row_difference <= TEMPORARY_OBSTACLE_MATCH_RADIUS_CELLS
        ):
            return existing_grid_cell

    return None


def add_or_refresh_temporary_obstacle(detected_grid_cell, current_simulation_time):
    matching_obstacle_cell = find_matching_temporary_obstacle(detected_grid_cell)

    if matching_obstacle_cell is not None:
        temporary_obstacles[matching_obstacle_cell] = current_simulation_time
        return matching_obstacle_cell

    temporary_obstacles[detected_grid_cell] = current_simulation_time
    return detected_grid_cell


def build_current_occupancy_grid(current_simulation_time):
    remove_expired_temporary_obstacles(current_simulation_time)
    combined_occupancy_grid = copy_occupancy_grid(static_occupancy_grid)

    for obstacle_grid_cell in temporary_obstacles.keys():
        mark_temporary_obstacle_on_grid(combined_occupancy_grid, obstacle_grid_cell)

    return combined_occupancy_grid


def grid_cell_is_static_obstacle(grid_cell):
    return static_occupancy_grid[grid_cell[0]][grid_cell[1]] == 1


# =========================================================
# A* AND PATH OPTIMIZATION
# =========================================================

def calculate_manhattan_distance(first_grid_cell, second_grid_cell):
    return (
        abs(first_grid_cell[0] - second_grid_cell[0])
        + abs(first_grid_cell[1] - second_grid_cell[1])
    )


def calculate_astar_path(occupancy_grid, starting_grid_cell, goal_grid_cell):
    """A* with a closed set to avoid redundant cell expansion."""

    frontier_priority_queue = []
    starting_priority = calculate_manhattan_distance(starting_grid_cell, goal_grid_cell)
    heapq.heappush(frontier_priority_queue, (starting_priority, starting_grid_cell))

    previous_grid_cell = {}
    movement_cost_from_start = {starting_grid_cell: 0}
    closed_grid_cells = set()

    while frontier_priority_queue:
        _, current_grid_cell = heapq.heappop(frontier_priority_queue)

        if current_grid_cell in closed_grid_cells:
            continue

        if current_grid_cell == goal_grid_cell:
            calculated_path = [current_grid_cell]
            while current_grid_cell in previous_grid_cell:
                current_grid_cell = previous_grid_cell[current_grid_cell]
                calculated_path.append(current_grid_cell)
            calculated_path.reverse()
            return calculated_path

        closed_grid_cells.add(current_grid_cell)
        current_grid_column, current_grid_row = current_grid_cell

        neighboring_grid_cells = [
            (current_grid_column + 1, current_grid_row),
            (current_grid_column - 1, current_grid_row),
            (current_grid_column, current_grid_row + 1),
            (current_grid_column, current_grid_row - 1),
        ]

        for neighboring_grid_cell in neighboring_grid_cells:
            neighboring_grid_column, neighboring_grid_row = neighboring_grid_cell

            if not (
                0 <= neighboring_grid_column < GRID_CELL_COUNT
                and 0 <= neighboring_grid_row < GRID_CELL_COUNT
            ):
                continue

            if occupancy_grid[neighboring_grid_column][neighboring_grid_row] == 1:
                continue

            if neighboring_grid_cell in closed_grid_cells:
                continue

            new_movement_cost_from_start = movement_cost_from_start[current_grid_cell] + 1

            if (
                neighboring_grid_cell not in movement_cost_from_start
                or new_movement_cost_from_start
                < movement_cost_from_start[neighboring_grid_cell]
            ):
                previous_grid_cell[neighboring_grid_cell] = current_grid_cell
                movement_cost_from_start[neighboring_grid_cell] = new_movement_cost_from_start

                estimated_total_path_cost = (
                    new_movement_cost_from_start
                    + calculate_manhattan_distance(neighboring_grid_cell, goal_grid_cell)
                )

                heapq.heappush(
                    frontier_priority_queue,
                    (estimated_total_path_cost, neighboring_grid_cell),
                )

    return None


def simplify_grid_path(grid_path):
    """Keep only start, turns, and final cell on straight A* runs."""

    if grid_path is None or len(grid_path) <= 2:
        return grid_path

    simplified_path = [grid_path[0]]
    previous_direction = None

    for path_index in range(1, len(grid_path)):
        previous_grid_cell = grid_path[path_index - 1]
        current_grid_cell = grid_path[path_index]
        current_direction = (
            current_grid_cell[0] - previous_grid_cell[0],
            current_grid_cell[1] - previous_grid_cell[1],
        )

        if previous_direction is not None and current_direction != previous_direction:
            simplified_path.append(previous_grid_cell)

        previous_direction = current_direction

    simplified_path.append(grid_path[-1])
    return simplified_path


def create_navigation_waypoints(planned_path, exact_goal_world_position):
    if planned_path is None:
        return []

    if len(planned_path) == 1:
        return [exact_goal_world_position]

    navigation_waypoints = [
        convert_grid_cell_to_world_position(path_grid_cell[0], path_grid_cell[1])
        for path_grid_cell in planned_path[1:]
    ]

    navigation_waypoints[-1] = exact_goal_world_position
    return navigation_waypoints


def calculate_route(
    start_world_x,
    start_world_y,
    goal_world_position,
    current_simulation_time,
):
    current_occupancy_grid = build_current_occupancy_grid(current_simulation_time)
    starting_grid_cell = convert_world_position_to_grid_cell(start_world_x, start_world_y)
    goal_grid_cell = convert_world_position_to_grid_cell(
        goal_world_position[0], goal_world_position[1]
    )

    # Obstacle inflation may overlap the cell currently occupied by the robot.
    current_occupancy_grid[starting_grid_cell[0]][starting_grid_cell[1]] = 0

    raw_grid_path = calculate_astar_path(
        current_occupancy_grid,
        starting_grid_cell,
        goal_grid_cell,
    )
    simplified_grid_path = simplify_grid_path(raw_grid_path)
    navigation_waypoints = create_navigation_waypoints(
        simplified_grid_path,
        goal_world_position,
    )

    return (
        raw_grid_path,
        simplified_grid_path,
        navigation_waypoints,
        starting_grid_cell,
        goal_grid_cell,
    )


# =========================================================
# SENSOR / OBSTACLE LOCATION FUNCTIONS
# =========================================================

def convert_sensor_value_to_meters(sensor_value):
    return sensor_value / 1000.0


def find_closest_detected_obstacle(
    front_sensor_value,
    left_sensor_value,
    right_sensor_value,
    detection_threshold,
):
    possible_detections = []

    sensor_definitions = [
        (
            "FRONT",
            front_sensor_value,
            FRONT_SENSOR_LOCAL_X,
            FRONT_SENSOR_LOCAL_Y,
            FRONT_SENSOR_ANGLE,
        ),
        (
            "LEFT",
            left_sensor_value,
            LEFT_SENSOR_LOCAL_X,
            LEFT_SENSOR_LOCAL_Y,
            LEFT_SENSOR_ANGLE,
        ),
        (
            "RIGHT",
            right_sensor_value,
            RIGHT_SENSOR_LOCAL_X,
            RIGHT_SENSOR_LOCAL_Y,
            RIGHT_SENSOR_ANGLE,
        ),
    ]

    for sensor_name, sensor_value, local_x, local_y, sensor_angle in sensor_definitions:
        if sensor_value < detection_threshold:
            possible_detections.append(
                {
                    "sensor_name": sensor_name,
                    "sensor_value": sensor_value,
                    "local_x": local_x,
                    "local_y": local_y,
                    "sensor_angle": sensor_angle,
                }
            )

    if not possible_detections:
        return None

    return min(possible_detections, key=lambda detection: detection["sensor_value"])


def estimate_obstacle_world_position(
    robot_world_x_position,
    robot_world_y_position,
    robot_current_yaw,
    obstacle_detection,
):
    sensor_local_x = obstacle_detection["local_x"]
    sensor_local_y = obstacle_detection["local_y"]
    obstacle_distance_meters = convert_sensor_value_to_meters(
        obstacle_detection["sensor_value"]
    )

    sensor_world_x = (
        robot_world_x_position
        + sensor_local_x * math.cos(robot_current_yaw)
        - sensor_local_y * math.sin(robot_current_yaw)
    )
    sensor_world_y = (
        robot_world_y_position
        + sensor_local_x * math.sin(robot_current_yaw)
        + sensor_local_y * math.cos(robot_current_yaw)
    )
    sensor_world_heading = robot_current_yaw + obstacle_detection["sensor_angle"]

    estimated_obstacle_world_x = (
        sensor_world_x + obstacle_distance_meters * math.cos(sensor_world_heading)
    )
    estimated_obstacle_world_y = (
        sensor_world_y + obstacle_distance_meters * math.sin(sensor_world_heading)
    )

    return estimated_obstacle_world_x, estimated_obstacle_world_y


# =========================================================
# STATIC MAP CREATION
# =========================================================

static_occupancy_grid = create_empty_occupancy_grid()

mark_rectangular_obstacle_on_grid(
    static_occupancy_grid,
    BOX_ONE_CENTER_X,
    BOX_ONE_CENTER_Y,
    BOX_ONE_SIZE_X,
    BOX_ONE_SIZE_Y,
    STATIC_OBSTACLE_SAFETY_MARGIN,
)

mark_rectangular_obstacle_on_grid(
    static_occupancy_grid,
    BOX_TWO_CENTER_X,
    BOX_TWO_CENTER_Y,
    BOX_TWO_SIZE_X,
    BOX_TWO_SIZE_Y,
    STATIC_OBSTACLE_SAFETY_MARGIN,
)


# =========================================================
# PERFORMANCE COUNTERS AND PATROL INITIALIZATION
# =========================================================

path_plan_count = 0
dynamic_replan_count = 0
heading_correction_count = 0
patrol_points_reached_count = 0
total_distance_traveled_meters = 0.0
previous_robot_world_position = None

current_patrol_location_index = 0
current_patrol_goal_world_position = PATROL_LOCATIONS[current_patrol_location_index]

(
    raw_grid_path,
    planned_grid_path,
    navigation_waypoints,
    starting_grid_cell,
    goal_grid_cell,
) = calculate_route(
    ROBOT_START_WORLD_POSITION[0],
    ROBOT_START_WORLD_POSITION[1],
    current_patrol_goal_world_position,
    amason_robot.getTime(),
)

path_plan_count += 1
current_waypoint_index = 0
navigation_state = "TURN_TO_WAYPOINT" if navigation_waypoints else "WAIT_FOR_PATH"
planning_reason = "INITIAL"
obstacle_confirmation_start_time = 0.0
last_path_retry_time = 0.0


# =========================================================
# CSV LOGGING SETUP
# =========================================================

navigation_log_file = open(LOG_FILE_PATH, "w", newline="")
navigation_log_writer = csv.writer(navigation_log_file)
navigation_log_writer.writerow(
    [
        "Time",
        "X",
        "Y",
        "Yaw",
        "LeftSensor",
        "FrontSensor",
        "RightSensor",
        "State",
        "WaypointNumber",
        "WaypointCount",
        "PatrolPointNumber",
        "PatrolTargetX",
        "PatrolTargetY",
        "TemporaryObstacleCount",
        "PathPlanCount",
        "DynamicReplanCount",
        "HeadingCorrectionCount",
        "PatrolPointsReached",
        "DistanceTraveledMeters",
        "Event",
        "EventX",
        "EventY",
    ]
)

last_log_time = -LOG_INTERVAL_SECONDS
last_console_status_time = -CONSOLE_STATUS_INTERVAL_SECONDS
last_log_flush_time = 0.0
pending_log_events = []
pending_event_world_x = ""
pending_event_world_y = ""


# =========================================================
# INITIAL OUTPUT
# =========================================================

print()
print("==========================================")
print("A.M.A.S.O.N. AUTONOMOUS PATROL")
print("==========================================")
print("Start location:", ROBOT_START_WORLD_POSITION)
print("Patrol point:", current_patrol_location_index + 1, current_patrol_goal_world_position)

if raw_grid_path is None:
    print("ERROR: A* COULD NOT FIND AN INITIAL PATH")
else:
    print("Raw A* cells:", len(raw_grid_path))
    print("Simplified path cells:", len(planned_grid_path))
    print("Navigation waypoints:", len(navigation_waypoints))

print("CSV log:", LOG_FILE_PATH)
print("==========================================")
print()


# =========================================================
# MAIN CONTROL LOOP
# =========================================================

while amason_robot.step(SIMULATION_TIME_STEP) != -1:
    current_simulation_time = amason_robot.getTime()

    current_gps_position = gps_sensor.getValues()
    robot_world_x_position = current_gps_position[0]
    robot_world_y_position = current_gps_position[1]

    if previous_robot_world_position is not None:
        movement_delta_x = robot_world_x_position - previous_robot_world_position[0]
        movement_delta_y = robot_world_y_position - previous_robot_world_position[1]
        total_distance_traveled_meters += math.sqrt(
            movement_delta_x ** 2 + movement_delta_y ** 2
        )

    previous_robot_world_position = (
        robot_world_x_position,
        robot_world_y_position,
    )

    robot_current_yaw = inertial_orientation_sensor.getRollPitchYaw()[2]

    front_obstacle_distance_value = front_distance_sensor.getValue()
    left_obstacle_distance_value = left_front_distance_sensor.getValue()
    right_obstacle_distance_value = right_front_distance_sensor.getValue()

    # -----------------------------------------------------
    # Detect and classify the closest obstacle.
    # -----------------------------------------------------

    closest_obstacle_detection = find_closest_detected_obstacle(
        front_obstacle_distance_value,
        left_obstacle_distance_value,
        right_obstacle_distance_value,
        OBSTACLE_DETECTION_THRESHOLD,
    )

    detected_obstacle_grid_cell = None
    matching_temporary_obstacle_cell = None
    detection_is_known_static_obstacle = False

    if closest_obstacle_detection is not None:
        estimated_obstacle_world_x, estimated_obstacle_world_y = (
            estimate_obstacle_world_position(
                robot_world_x_position,
                robot_world_y_position,
                robot_current_yaw,
                closest_obstacle_detection,
            )
        )

        detected_obstacle_grid_cell = convert_world_position_to_grid_cell(
            estimated_obstacle_world_x,
            estimated_obstacle_world_y,
        )

        detection_is_known_static_obstacle = grid_cell_is_static_obstacle(
            detected_obstacle_grid_cell
        )
        matching_temporary_obstacle_cell = find_matching_temporary_obstacle(
            detected_obstacle_grid_cell
        )

        if matching_temporary_obstacle_cell is not None:
            temporary_obstacles[matching_temporary_obstacle_cell] = current_simulation_time

    # Only a genuinely new obstacle interrupts forward motion.
    if (
        navigation_state == "DRIVE_TO_WAYPOINT"
        and closest_obstacle_detection is not None
        and not detection_is_known_static_obstacle
        and matching_temporary_obstacle_cell is None
    ):
        left_wheel_motor.setVelocity(0.0)
        right_wheel_motor.setVelocity(0.0)
        obstacle_confirmation_start_time = current_simulation_time
        navigation_state = "CONFIRM_OBSTACLE"
        pending_log_events.append("OBSTACLE_DETECTED")
        print(
            "Unexpected obstacle detected by",
            closest_obstacle_detection["sensor_name"],
            "sensor at grid cell",
            detected_obstacle_grid_cell,
        )

    # =====================================================
    # FINITE STATE MACHINE
    # =====================================================

    if navigation_state == "CONFIRM_OBSTACLE":
        left_wheel_motor.setVelocity(0.0)
        right_wheel_motor.setVelocity(0.0)

        confirmation_obstacle_detection = find_closest_detected_obstacle(
            front_obstacle_distance_value,
            left_obstacle_distance_value,
            right_obstacle_distance_value,
            OBSTACLE_CLEAR_THRESHOLD,
        )

        if confirmation_obstacle_detection is None:
            navigation_state = "TURN_TO_WAYPOINT"
            pending_log_events.append("OBSTACLE_CLEARED")
            print("Obstacle cleared before confirmation.")

        elif (
            current_simulation_time - obstacle_confirmation_start_time
            >= OBSTACLE_CONFIRMATION_TIME_SECONDS
        ):
            estimated_obstacle_world_x, estimated_obstacle_world_y = (
                estimate_obstacle_world_position(
                    robot_world_x_position,
                    robot_world_y_position,
                    robot_current_yaw,
                    confirmation_obstacle_detection,
                )
            )

            detected_obstacle_grid_cell = convert_world_position_to_grid_cell(
                estimated_obstacle_world_x,
                estimated_obstacle_world_y,
            )

            if grid_cell_is_static_obstacle(detected_obstacle_grid_cell):
                navigation_state = "TURN_TO_WAYPOINT"
                pending_log_events.append("STATIC_OBSTACLE_RECOGNIZED")

            else:
                matching_temporary_obstacle_cell = find_matching_temporary_obstacle(
                    detected_obstacle_grid_cell
                )

                if matching_temporary_obstacle_cell is not None:
                    temporary_obstacles[matching_temporary_obstacle_cell] = (
                        current_simulation_time
                    )
                    navigation_state = "TURN_TO_WAYPOINT"
                    pending_log_events.append("KNOWN_TEMPORARY_OBSTACLE")

                else:
                    stored_obstacle_grid_cell = add_or_refresh_temporary_obstacle(
                        detected_obstacle_grid_cell,
                        current_simulation_time,
                    )
                    dynamic_replan_count += 1
                    planning_reason = "DYNAMIC_OBSTACLE"
                    navigation_state = "PLAN_ROUTE"
                    pending_log_events.append("TEMPORARY_OBSTACLE_ADDED")
                    pending_event_world_x = estimated_obstacle_world_x
                    pending_event_world_y = estimated_obstacle_world_y

                    print()
                    print("==========================================")
                    print("NEW TEMPORARY OBSTACLE ADDED")
                    print("Grid cell:", stored_obstacle_grid_cell)
                    print(
                        "Estimated position:",
                        (
                            round(estimated_obstacle_world_x, 3),
                            round(estimated_obstacle_world_y, 3),
                        ),
                    )
                    print("Replanning patrol route...")
                    print("==========================================")
                    print()

    elif navigation_state == "PLAN_ROUTE":
        left_wheel_motor.setVelocity(0.0)
        right_wheel_motor.setVelocity(0.0)

        (
            raw_grid_path,
            planned_grid_path,
            navigation_waypoints,
            starting_grid_cell,
            goal_grid_cell,
        ) = calculate_route(
            robot_world_x_position,
            robot_world_y_position,
            current_patrol_goal_world_position,
            current_simulation_time,
        )

        path_plan_count += 1
        current_waypoint_index = 0

        if raw_grid_path is None or not navigation_waypoints:
            last_path_retry_time = current_simulation_time
            navigation_state = "WAIT_FOR_PATH"
            pending_log_events.append("NO_PATH")
            print("No route currently available. A.M.A.S.O.N. will retry.")
        else:
            navigation_state = "TURN_TO_WAYPOINT"
            pending_log_events.append("PATH_PLANNED")

            print()
            print("==========================================")
            print("A* PATH PLANNED")
            print("Reason:", planning_reason)
            print(
                "Patrol target:",
                current_patrol_location_index + 1,
                current_patrol_goal_world_position,
            )
            print("Raw A* cells:", len(raw_grid_path))
            print("Simplified cells:", len(planned_grid_path))
            print("Navigation waypoints:", len(navigation_waypoints))
            print("==========================================")
            print()

    elif navigation_state == "WAIT_FOR_PATH":
        left_wheel_motor.setVelocity(0.0)
        right_wheel_motor.setVelocity(0.0)

        if (
            current_simulation_time - last_path_retry_time
            >= PATH_RETRY_INTERVAL_SECONDS
        ):
            last_path_retry_time = current_simulation_time
            planning_reason = "PATH_RETRY"
            navigation_state = "PLAN_ROUTE"

    elif navigation_state == "TURN_TO_WAYPOINT":
        if current_waypoint_index >= len(navigation_waypoints):
            navigation_state = "PATROL_POINT_REACHED"
        else:
            waypoint_target_x_position, waypoint_target_y_position = (
                navigation_waypoints[current_waypoint_index]
            )
            distance_to_waypoint_x = waypoint_target_x_position - robot_world_x_position
            distance_to_waypoint_y = waypoint_target_y_position - robot_world_y_position
            straight_line_distance_to_waypoint = math.sqrt(
                distance_to_waypoint_x ** 2 + distance_to_waypoint_y ** 2
            )

            if straight_line_distance_to_waypoint <= WAYPOINT_DISTANCE_TOLERANCE:
                current_waypoint_index += 1
                if current_waypoint_index >= len(navigation_waypoints):
                    navigation_state = "PATROL_POINT_REACHED"
            else:
                desired_waypoint_heading = math.atan2(
                    distance_to_waypoint_y,
                    distance_to_waypoint_x,
                )
                waypoint_heading_error = normalize_angle_radians(
                    desired_waypoint_heading - robot_current_yaw
                )

                if abs(waypoint_heading_error) <= WAYPOINT_HEADING_TOLERANCE:
                    left_wheel_motor.setVelocity(0.0)
                    right_wheel_motor.setVelocity(0.0)
                    navigation_state = "DRIVE_TO_WAYPOINT"
                elif waypoint_heading_error > 0:
                    left_wheel_motor.setVelocity(-TURNING_WHEEL_SPEED)
                    right_wheel_motor.setVelocity(TURNING_WHEEL_SPEED)
                else:
                    left_wheel_motor.setVelocity(TURNING_WHEEL_SPEED)
                    right_wheel_motor.setVelocity(-TURNING_WHEEL_SPEED)

    elif navigation_state == "DRIVE_TO_WAYPOINT":
        if current_waypoint_index >= len(navigation_waypoints):
            navigation_state = "PATROL_POINT_REACHED"
        else:
            waypoint_target_x_position, waypoint_target_y_position = (
                navigation_waypoints[current_waypoint_index]
            )
            distance_to_waypoint_x = waypoint_target_x_position - robot_world_x_position
            distance_to_waypoint_y = waypoint_target_y_position - robot_world_y_position
            straight_line_distance_to_waypoint = math.sqrt(
                distance_to_waypoint_x ** 2 + distance_to_waypoint_y ** 2
            )
            desired_waypoint_heading = math.atan2(
                distance_to_waypoint_y,
                distance_to_waypoint_x,
            )
            waypoint_heading_error = normalize_angle_radians(
                desired_waypoint_heading - robot_current_yaw
            )

            if straight_line_distance_to_waypoint <= WAYPOINT_DISTANCE_TOLERANCE:
                current_waypoint_index += 1
                left_wheel_motor.setVelocity(0.0)
                right_wheel_motor.setVelocity(0.0)

                if current_waypoint_index >= len(navigation_waypoints):
                    navigation_state = "PATROL_POINT_REACHED"
                else:
                    navigation_state = "TURN_TO_WAYPOINT"

            elif abs(waypoint_heading_error) > WAYPOINT_HEADING_TOLERANCE:
                heading_correction_count += 1
                left_wheel_motor.setVelocity(0.0)
                right_wheel_motor.setVelocity(0.0)
                navigation_state = "TURN_TO_WAYPOINT"

            else:
                left_wheel_motor.setVelocity(FORWARD_WHEEL_SPEED)
                right_wheel_motor.setVelocity(FORWARD_WHEEL_SPEED)

    elif navigation_state == "PATROL_POINT_REACHED":
        left_wheel_motor.setVelocity(0.0)
        right_wheel_motor.setVelocity(0.0)
        patrol_points_reached_count += 1
        pending_log_events.append("PATROL_POINT_REACHED")

        print()
        print(
            "Patrol point",
            current_patrol_location_index + 1,
            "reached:",
            current_patrol_goal_world_position,
        )

        current_patrol_location_index = (
            current_patrol_location_index + 1
        ) % len(PATROL_LOCATIONS)
        current_patrol_goal_world_position = PATROL_LOCATIONS[
            current_patrol_location_index
        ]
        planning_reason = "NEXT_PATROL_POINT"
        navigation_state = "PLAN_ROUTE"

        print(
            "Next patrol point:",
            current_patrol_location_index + 1,
            current_patrol_goal_world_position,
        )

    elif navigation_state == "STOPPED":
        left_wheel_motor.setVelocity(0.0)
        right_wheel_motor.setVelocity(0.0)

    # =====================================================
    # CSV DATA LOGGING
    # =====================================================

    if current_simulation_time - last_log_time >= LOG_INTERVAL_SECONDS:
        last_log_time = current_simulation_time

        if current_waypoint_index < len(navigation_waypoints):
            displayed_waypoint_number = current_waypoint_index + 1
        else:
            displayed_waypoint_number = len(navigation_waypoints)

        navigation_log_writer.writerow(
            [
                round(current_simulation_time, 3),
                robot_world_x_position,
                robot_world_y_position,
                robot_current_yaw,
                left_obstacle_distance_value,
                front_obstacle_distance_value,
                right_obstacle_distance_value,
                navigation_state,
                displayed_waypoint_number,
                len(navigation_waypoints),
                current_patrol_location_index + 1,
                current_patrol_goal_world_position[0],
                current_patrol_goal_world_position[1],
                len(temporary_obstacles),
                path_plan_count,
                dynamic_replan_count,
                heading_correction_count,
                patrol_points_reached_count,
                total_distance_traveled_meters,
                "|".join(pending_log_events),
                pending_event_world_x,
                pending_event_world_y,
            ]
        )

        pending_log_events.clear()
        pending_event_world_x = ""
        pending_event_world_y = ""

    if current_simulation_time - last_log_flush_time >= 1.0:
        navigation_log_file.flush()
        last_log_flush_time = current_simulation_time

    # =====================================================
    # REDUCED-RATE CONSOLE STATUS
    # =====================================================

    if (
        current_simulation_time - last_console_status_time
        >= CONSOLE_STATUS_INTERVAL_SECONDS
    ):
        last_console_status_time = current_simulation_time

        if current_waypoint_index < len(navigation_waypoints):
            current_target_description = navigation_waypoints[current_waypoint_index]
            waypoint_description = (
                f"{current_waypoint_index + 1}/{len(navigation_waypoints)}"
            )
        else:
            current_target_description = current_patrol_goal_world_position
            waypoint_description = (
                f"{len(navigation_waypoints)}/{len(navigation_waypoints)}"
            )

        print(
            f"X={robot_world_x_position:.3f}, "
            f"Y={robot_world_y_position:.3f}, "
            f"Yaw={robot_current_yaw:.3f}, "
            f"Patrol={current_patrol_location_index + 1}/{len(PATROL_LOCATIONS)}, "
            f"Waypoint={waypoint_description}, "
            f"Target={current_target_description}, "
            f"TempObstacles={len(temporary_obstacles)}, "
            f"Replans={dynamic_replan_count}, "
            f"Distance={total_distance_traveled_meters:.2f}m, "
            f"State={navigation_state}"
        )


# =========================================================
# CLEAN SHUTDOWN
# =========================================================

left_wheel_motor.setVelocity(0.0)
right_wheel_motor.setVelocity(0.0)
navigation_log_file.flush()
navigation_log_file.close()

print()
print("A.M.A.S.O.N. controller stopped.")
print("Navigation log saved to:", LOG_FILE_PATH)
