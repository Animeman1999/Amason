"""A.M.A.S.O.N. Webots Controller

Autonomous Mobile AI System for Optimized Navigation

Current controller features:
- autonomous patrol task execution
- simplified A* waypoint paths
- A* closed-set optimization
- dynamic obstacle memory and replanning
- simulated noisy GPS and distance-sensor measurements
- grid-based Markov/Bayesian localization
- occupancy-grid observation updates from movement and sensors
- local reactive BACKTRACK / ESCAPE_TURN behavior
- reduced-rate console output
- CSV performance/data logging
"""

from controller import Robot
import csv
import heapq
import math
import os
import random


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

# ---------------------------------------------------------
# LOCAL REACTIVE AVOIDANCE SETTINGS
# ---------------------------------------------------------

# If an obstacle is this close, immediately use a local
# escape behavior instead of waiting for global replanning.
EMERGENCY_OBSTACLE_THRESHOLD = 120
BACKTRACK_WHEEL_SPEED = 1.5
BACKTRACK_DURATION_SECONDS = 1.50
ESCAPE_TURN_ANGLE_RADIANS = math.radians(45)
NO_PATH_ATTEMPTS_BEFORE_BACKTRACK = 2

# Wheel geometry used by the localization motion model.
WHEEL_RADIUS_METERS = 0.075
WHEEL_SEPARATION_METERS = 0.44

# ---------------------------------------------------------
# SIMULATED SENSOR NOISE / LOCALIZATION SETTINGS
# ---------------------------------------------------------

# A fixed seed makes demonstrations repeatable. Change this
# number to obtain a different noise sequence.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

GPS_NOISE_STANDARD_DEVIATION_METERS = 0.06
DISTANCE_SENSOR_NOISE_STANDARD_DEVIATION = 8.0
LOCALIZATION_GPS_LIKELIHOOD_STANDARD_DEVIATION_METERS = 0.18
LOCALIZATION_INITIAL_STANDARD_DEVIATION_METERS = 0.20
LOCALIZATION_UPDATE_INTERVAL_SECONDS = 0.10

# The simulated distance sensors use the current lookup table
# where approximately 1000 corresponds to one meter.
DISTANCE_SENSOR_MINIMUM_VALUE = 0.0
DISTANCE_SENSOR_MAXIMUM_VALUE = 1000.0
MAPPING_OBSTACLE_SENSOR_THRESHOLD = 950.0


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
# SIMULATED NOISE FUNCTIONS
# =========================================================

def add_gaussian_noise(value, standard_deviation, minimum_value=None, maximum_value=None):
    noisy_value = value + random.gauss(0.0, standard_deviation)

    if minimum_value is not None:
        noisy_value = max(minimum_value, noisy_value)

    if maximum_value is not None:
        noisy_value = min(maximum_value, noisy_value)

    return noisy_value


def create_noisy_gps_measurement(actual_world_x, actual_world_y):
    noisy_world_x = add_gaussian_noise(
        actual_world_x,
        GPS_NOISE_STANDARD_DEVIATION_METERS,
        ARENA_MINIMUM_COORDINATE,
        ARENA_MINIMUM_COORDINATE + ARENA_SIZE_METERS,
    )
    noisy_world_y = add_gaussian_noise(
        actual_world_y,
        GPS_NOISE_STANDARD_DEVIATION_METERS,
        ARENA_MINIMUM_COORDINATE,
        ARENA_MINIMUM_COORDINATE + ARENA_SIZE_METERS,
    )
    return noisy_world_x, noisy_world_y


def create_noisy_distance_sensor_measurement(actual_sensor_value):
    return add_gaussian_noise(
        actual_sensor_value,
        DISTANCE_SENSOR_NOISE_STANDARD_DEVIATION,
        DISTANCE_SENSOR_MINIMUM_VALUE,
        DISTANCE_SENSOR_MAXIMUM_VALUE,
    )


# =========================================================
# MARKOV / BAYESIAN LOCALIZATION FUNCTIONS
# =========================================================

def create_probability_grid():
    return [
        [0.0 for _ in range(GRID_CELL_COUNT)]
        for _ in range(GRID_CELL_COUNT)
    ]


def normalize_probability_grid(probability_grid):
    probability_total = sum(
        probability_grid[grid_column][grid_row]
        for grid_column in range(GRID_CELL_COUNT)
        for grid_row in range(GRID_CELL_COUNT)
    )

    if probability_total <= 0.0:
        return False

    for grid_column in range(GRID_CELL_COUNT):
        for grid_row in range(GRID_CELL_COUNT):
            probability_grid[grid_column][grid_row] /= probability_total

    return True


def initialize_localization_probability_grid(initial_world_x, initial_world_y):
    probability_grid = create_probability_grid()
    variance = LOCALIZATION_INITIAL_STANDARD_DEVIATION_METERS ** 2

    for grid_column in range(GRID_CELL_COUNT):
        for grid_row in range(GRID_CELL_COUNT):
            if localization_occupancy_grid[grid_column][grid_row] == 1:
                continue

            cell_world_x, cell_world_y = convert_grid_cell_to_world_position(
                grid_column,
                grid_row,
            )
            squared_distance = (
                (cell_world_x - initial_world_x) ** 2
                + (cell_world_y - initial_world_y) ** 2
            )
            probability_grid[grid_column][grid_row] = math.exp(
                -squared_distance / (2.0 * variance)
            )

    normalize_probability_grid(probability_grid)
    return probability_grid


def determine_cardinal_motion_direction(robot_yaw, commanded_distance_meters):
    movement_heading = robot_yaw
    if commanded_distance_meters < 0.0:
        movement_heading = normalize_angle_radians(robot_yaw + math.pi)

    heading_x = math.cos(movement_heading)
    heading_y = math.sin(movement_heading)

    if abs(heading_x) >= abs(heading_y):
        return (1, 0) if heading_x >= 0.0 else (-1, 0)

    return (0, 1) if heading_y >= 0.0 else (0, -1)


def apply_markov_motion_update(
    probability_grid,
    commanded_distance_meters,
    robot_yaw,
):
    """Prediction step for grid-based Markov localization.

    The belief is shifted probabilistically in the commanded
    direction. Some probability remains in the current cell
    or drifts laterally to represent wheel slip and motion error.
    """

    if abs(commanded_distance_meters) < 0.0001:
        return probability_grid

    predicted_probability_grid = create_probability_grid()
    intended_column_offset, intended_row_offset = determine_cardinal_motion_direction(
        robot_yaw,
        commanded_distance_meters,
    )

    lateral_offsets = [
        (-intended_row_offset, intended_column_offset),
        (intended_row_offset, -intended_column_offset),
    ]

    cell_movement_fraction = min(
        1.0,
        abs(commanded_distance_meters) / GRID_CELL_SIZE_METERS,
    )

    intended_move_probability = 0.80 * cell_movement_fraction
    lateral_move_probability = 0.05 * cell_movement_fraction
    stay_probability = (
        1.0
        - intended_move_probability
        - 2.0 * lateral_move_probability
    )

    def add_transition_probability(
        source_probability,
        source_column,
        source_row,
        column_offset,
        row_offset,
        transition_probability,
    ):
        destination_column = source_column + column_offset
        destination_row = source_row + row_offset

        if (
            0 <= destination_column < GRID_CELL_COUNT
            and 0 <= destination_row < GRID_CELL_COUNT
            and localization_occupancy_grid[destination_column][destination_row] == 0
        ):
            predicted_probability_grid[destination_column][destination_row] += (
                source_probability * transition_probability
            )
        else:
            predicted_probability_grid[source_column][source_row] += (
                source_probability * transition_probability
            )

    for grid_column in range(GRID_CELL_COUNT):
        for grid_row in range(GRID_CELL_COUNT):
            source_probability = probability_grid[grid_column][grid_row]
            if source_probability <= 0.0:
                continue

            predicted_probability_grid[grid_column][grid_row] += (
                source_probability * stay_probability
            )

            add_transition_probability(
                source_probability,
                grid_column,
                grid_row,
                intended_column_offset,
                intended_row_offset,
                intended_move_probability,
            )

            for lateral_column_offset, lateral_row_offset in lateral_offsets:
                add_transition_probability(
                    source_probability,
                    grid_column,
                    grid_row,
                    lateral_column_offset,
                    lateral_row_offset,
                    lateral_move_probability,
                )

    normalize_probability_grid(predicted_probability_grid)
    return predicted_probability_grid


def apply_gps_measurement_update(
    probability_grid,
    noisy_gps_world_x,
    noisy_gps_world_y,
):
    """Bayesian correction step using a noisy GPS measurement."""

    corrected_probability_grid = create_probability_grid()
    variance = LOCALIZATION_GPS_LIKELIHOOD_STANDARD_DEVIATION_METERS ** 2

    for grid_column in range(GRID_CELL_COUNT):
        for grid_row in range(GRID_CELL_COUNT):
            if localization_occupancy_grid[grid_column][grid_row] == 1:
                continue

            cell_world_x, cell_world_y = convert_grid_cell_to_world_position(
                grid_column,
                grid_row,
            )
            squared_measurement_error = (
                (cell_world_x - noisy_gps_world_x) ** 2
                + (cell_world_y - noisy_gps_world_y) ** 2
            )
            measurement_likelihood = math.exp(
                -squared_measurement_error / (2.0 * variance)
            )

            corrected_probability_grid[grid_column][grid_row] = (
                probability_grid[grid_column][grid_row]
                * max(measurement_likelihood, 1e-12)
            )

    if not normalize_probability_grid(corrected_probability_grid):
        return initialize_localization_probability_grid(
            noisy_gps_world_x,
            noisy_gps_world_y,
        )

    return corrected_probability_grid


def estimate_position_from_probability_grid(probability_grid):
    estimated_world_x = 0.0
    estimated_world_y = 0.0
    most_likely_grid_cell = (0, 0)
    highest_cell_probability = -1.0

    for grid_column in range(GRID_CELL_COUNT):
        for grid_row in range(GRID_CELL_COUNT):
            cell_probability = probability_grid[grid_column][grid_row]
            cell_world_x, cell_world_y = convert_grid_cell_to_world_position(
                grid_column,
                grid_row,
            )

            estimated_world_x += cell_probability * cell_world_x
            estimated_world_y += cell_probability * cell_world_y

            if cell_probability > highest_cell_probability:
                highest_cell_probability = cell_probability
                most_likely_grid_cell = (grid_column, grid_row)

    return (
        estimated_world_x,
        estimated_world_y,
        most_likely_grid_cell,
        highest_cell_probability,
    )


# =========================================================
# OBSERVED OCCUPANCY GRID FUNCTIONS
# =========================================================

def create_observed_occupancy_grid():
    """Create an observation map: -1 unknown, 0 free, 1 occupied."""
    return [
        [-1 for _ in range(GRID_CELL_COUNT)]
        for _ in range(GRID_CELL_COUNT)
    ]


def mark_robot_position_as_observed_free(
    observed_grid,
    robot_world_x_position,
    robot_world_y_position,
):
    grid_column, grid_row = convert_world_position_to_grid_cell(
        robot_world_x_position,
        robot_world_y_position,
    )

    if localization_occupancy_grid[grid_column][grid_row] == 0:
        observed_grid[grid_column][grid_row] = 0


def calculate_sensor_world_geometry(
    robot_world_x_position,
    robot_world_y_position,
    robot_current_yaw,
    sensor_local_x,
    sensor_local_y,
    sensor_angle,
):
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
    sensor_world_heading = robot_current_yaw + sensor_angle
    return sensor_world_x, sensor_world_y, sensor_world_heading


def update_observed_map_from_sensor(
    observed_grid,
    robot_world_x_position,
    robot_world_y_position,
    robot_current_yaw,
    sensor_local_x,
    sensor_local_y,
    sensor_angle,
    noisy_sensor_value,
):
    """Update occupancy observations along one distance-sensor ray."""

    sensor_world_x, sensor_world_y, sensor_world_heading = (
        calculate_sensor_world_geometry(
            robot_world_x_position,
            robot_world_y_position,
            robot_current_yaw,
            sensor_local_x,
            sensor_local_y,
            sensor_angle,
        )
    )

    measured_distance_meters = min(
        1.0,
        max(0.0, noisy_sensor_value / 1000.0),
    )
    ray_end_x = sensor_world_x + measured_distance_meters * math.cos(
        sensor_world_heading
    )
    ray_end_y = sensor_world_y + measured_distance_meters * math.sin(
        sensor_world_heading
    )

    ray_length = math.sqrt(
        (ray_end_x - sensor_world_x) ** 2
        + (ray_end_y - sensor_world_y) ** 2
    )
    number_of_samples = max(
        1,
        int(ray_length / (GRID_CELL_SIZE_METERS / 2.0)),
    )

    ray_grid_cells = []
    for sample_index in range(number_of_samples + 1):
        interpolation_fraction = sample_index / number_of_samples
        sample_world_x = sensor_world_x + (
            ray_end_x - sensor_world_x
        ) * interpolation_fraction
        sample_world_y = sensor_world_y + (
            ray_end_y - sensor_world_y
        ) * interpolation_fraction
        sample_grid_cell = convert_world_position_to_grid_cell(
            sample_world_x,
            sample_world_y,
        )
        if not ray_grid_cells or ray_grid_cells[-1] != sample_grid_cell:
            ray_grid_cells.append(sample_grid_cell)

    obstacle_detected = noisy_sensor_value < MAPPING_OBSTACLE_SENSOR_THRESHOLD

    for ray_index, (grid_column, grid_row) in enumerate(ray_grid_cells):
        is_last_cell = ray_index == len(ray_grid_cells) - 1

        if is_last_cell and obstacle_detected:
            observed_grid[grid_column][grid_row] = 1
        elif localization_occupancy_grid[grid_column][grid_row] == 0:
            observed_grid[grid_column][grid_row] = 0


def count_observed_map_cells(observed_grid):
    return sum(
        1
        for grid_column in range(GRID_CELL_COUNT)
        for grid_row in range(GRID_CELL_COUNT)
        if observed_grid[grid_column][grid_row] != -1
    )


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

    sensor_world_x, sensor_world_y, sensor_world_heading = (
        calculate_sensor_world_geometry(
            robot_world_x_position,
            robot_world_y_position,
            robot_current_yaw,
            sensor_local_x,
            sensor_local_y,
            obstacle_detection["sensor_angle"],
        )
    )

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

# Localization uses physical obstacle footprints rather than
# the larger planning safety margins.
localization_occupancy_grid = create_empty_occupancy_grid()
mark_rectangular_obstacle_on_grid(
    localization_occupancy_grid,
    BOX_ONE_CENTER_X,
    BOX_ONE_CENTER_Y,
    BOX_ONE_SIZE_X,
    BOX_ONE_SIZE_Y,
    0.0,
)
mark_rectangular_obstacle_on_grid(
    localization_occupancy_grid,
    BOX_TWO_CENTER_X,
    BOX_TWO_CENTER_Y,
    BOX_TWO_SIZE_X,
    BOX_TWO_SIZE_Y,
    0.0,
)

observed_occupancy_grid = create_observed_occupancy_grid()
for initialization_grid_column in range(GRID_CELL_COUNT):
    for initialization_grid_row in range(GRID_CELL_COUNT):
        if localization_occupancy_grid[initialization_grid_column][initialization_grid_row] == 1:
            observed_occupancy_grid[initialization_grid_column][initialization_grid_row] = 1

localization_probability_grid = initialize_localization_probability_grid(
    ROBOT_START_WORLD_POSITION[0],
    ROBOT_START_WORLD_POSITION[1],
)
(
    localized_robot_world_x_position,
    localized_robot_world_y_position,
    localized_robot_grid_cell,
    localization_confidence,
) = estimate_position_from_probability_grid(localization_probability_grid)


# =========================================================
# PERFORMANCE COUNTERS AND PATROL INITIALIZATION
# =========================================================

path_plan_count = 0
dynamic_replan_count = 0
heading_correction_count = 0
patrol_points_reached_count = 0
backtrack_count = 0
escape_turn_count = 0
total_distance_traveled_meters = 0.0
previous_actual_robot_world_position = None

consecutive_no_path_count = 0
backtrack_start_time = 0.0
backtrack_reason = ""
escape_turn_target_yaw = 0.0
previous_control_loop_time = amason_robot.getTime()
last_localization_update_time = amason_robot.getTime()
commanded_linear_distance_since_localization_update = 0.0

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
        "ActualX",
        "ActualY",
        "NoisyGpsX",
        "NoisyGpsY",
        "LocalizedX",
        "LocalizedY",
        "LocalizedGridColumn",
        "LocalizedGridRow",
        "LocalizationConfidence",
        "ObservedMapCellCount",
        "BacktrackCount",
        "EscapeTurnCount",
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
print("Localization: Markov prediction + Bayesian noisy-GPS correction")
print("GPS noise sigma:", GPS_NOISE_STANDARD_DEVIATION_METERS, "meters")
print("Distance sensor noise sigma:", DISTANCE_SENSOR_NOISE_STANDARD_DEVIATION)
print("Local escape behavior: BACKTRACK -> ESCAPE_TURN -> PLAN_ROUTE")
print("==========================================")
print()


# =========================================================
# MAIN CONTROL LOOP
# =========================================================

while amason_robot.step(SIMULATION_TIME_STEP) != -1:
    current_simulation_time = amason_robot.getTime()
    control_loop_delta_time = max(
        0.0,
        current_simulation_time - previous_control_loop_time,
    )
    previous_control_loop_time = current_simulation_time

    # -----------------------------------------------------
    # SIMULATOR GROUND TRUTH
    # -----------------------------------------------------
    # Ground-truth GPS is retained for evaluation/logging.
    # Navigation and mapping use the Markov-localized position.

    current_actual_gps_position = gps_sensor.getValues()
    actual_robot_world_x_position = current_actual_gps_position[0]
    actual_robot_world_y_position = current_actual_gps_position[1]

    if previous_actual_robot_world_position is not None:
        movement_delta_x = (
            actual_robot_world_x_position
            - previous_actual_robot_world_position[0]
        )
        movement_delta_y = (
            actual_robot_world_y_position
            - previous_actual_robot_world_position[1]
        )
        total_distance_traveled_meters += math.sqrt(
            movement_delta_x ** 2 + movement_delta_y ** 2
        )

    previous_actual_robot_world_position = (
        actual_robot_world_x_position,
        actual_robot_world_y_position,
    )

    robot_current_yaw = inertial_orientation_sensor.getRollPitchYaw()[2]

    # -----------------------------------------------------
    # SIMULATED NOISY SENSOR DATA
    # -----------------------------------------------------

    noisy_gps_world_x, noisy_gps_world_y = create_noisy_gps_measurement(
        actual_robot_world_x_position,
        actual_robot_world_y_position,
    )

    actual_front_sensor_value = front_distance_sensor.getValue()
    actual_left_sensor_value = left_front_distance_sensor.getValue()
    actual_right_sensor_value = right_front_distance_sensor.getValue()

    front_obstacle_distance_value = create_noisy_distance_sensor_measurement(
        actual_front_sensor_value
    )
    left_obstacle_distance_value = create_noisy_distance_sensor_measurement(
        actual_left_sensor_value
    )
    right_obstacle_distance_value = create_noisy_distance_sensor_measurement(
        actual_right_sensor_value
    )

    # -----------------------------------------------------
    # MARKOV LOCALIZATION MOTION PREDICTION
    # -----------------------------------------------------

    commanded_left_wheel_speed = left_wheel_motor.getVelocity()
    commanded_right_wheel_speed = right_wheel_motor.getVelocity()
    commanded_linear_velocity = (
        WHEEL_RADIUS_METERS
        * (commanded_left_wheel_speed + commanded_right_wheel_speed)
        / 2.0
    )
    commanded_linear_distance_since_localization_update += (
        commanded_linear_velocity * control_loop_delta_time
    )

    if (
        current_simulation_time - last_localization_update_time
        >= LOCALIZATION_UPDATE_INTERVAL_SECONDS
    ):
        localization_probability_grid = apply_markov_motion_update(
            localization_probability_grid,
            commanded_linear_distance_since_localization_update,
            robot_current_yaw,
        )
        localization_probability_grid = apply_gps_measurement_update(
            localization_probability_grid,
            noisy_gps_world_x,
            noisy_gps_world_y,
        )
        (
            localized_robot_world_x_position,
            localized_robot_world_y_position,
            localized_robot_grid_cell,
            localization_confidence,
        ) = estimate_position_from_probability_grid(
            localization_probability_grid
        )

        commanded_linear_distance_since_localization_update = 0.0
        last_localization_update_time = current_simulation_time

    # The controller uses the probabilistic localization estimate.
    robot_world_x_position = localized_robot_world_x_position
    robot_world_y_position = localized_robot_world_y_position

    # -----------------------------------------------------
    # OCCUPANCY-MAP OBSERVATION UPDATE
    # -----------------------------------------------------

    mark_robot_position_as_observed_free(
        observed_occupancy_grid,
        robot_world_x_position,
        robot_world_y_position,
    )

    update_observed_map_from_sensor(
        observed_occupancy_grid,
        robot_world_x_position,
        robot_world_y_position,
        robot_current_yaw,
        FRONT_SENSOR_LOCAL_X,
        FRONT_SENSOR_LOCAL_Y,
        FRONT_SENSOR_ANGLE,
        front_obstacle_distance_value,
    )
    update_observed_map_from_sensor(
        observed_occupancy_grid,
        robot_world_x_position,
        robot_world_y_position,
        robot_current_yaw,
        LEFT_SENSOR_LOCAL_X,
        LEFT_SENSOR_LOCAL_Y,
        LEFT_SENSOR_ANGLE,
        left_obstacle_distance_value,
    )
    update_observed_map_from_sensor(
        observed_occupancy_grid,
        robot_world_x_position,
        robot_world_y_position,
        robot_current_yaw,
        RIGHT_SENSOR_LOCAL_X,
        RIGHT_SENSOR_LOCAL_Y,
        RIGHT_SENSOR_ANGLE,
        right_obstacle_distance_value,
    )

    remove_expired_temporary_obstacles(current_simulation_time)

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

    # -----------------------------------------------------
    # EMERGENCY LOCAL REACTIVE AVOIDANCE
    # -----------------------------------------------------
    # This is deliberately independent of A*. If the robot
    # gets very close to an object, it backs away first and
    # then turns toward the clearer side before replanning.

    closest_sensor_value = min(
        front_obstacle_distance_value,
        left_obstacle_distance_value,
        right_obstacle_distance_value,
    )

    if (
        navigation_state == "DRIVE_TO_WAYPOINT"
        and closest_sensor_value < EMERGENCY_OBSTACLE_THRESHOLD
    ):
        left_wheel_motor.setVelocity(0.0)
        right_wheel_motor.setVelocity(0.0)
        backtrack_start_time = current_simulation_time
        backtrack_reason = "EMERGENCY_OBSTACLE"
        backtrack_count += 1
        navigation_state = "BACKTRACK"
        pending_log_events.append("BACKTRACK_STARTED")
        print(
            "Emergency obstacle range detected. "
            "Starting local BACKTRACK behavior."
        )

    # Only a genuinely new obstacle interrupts ordinary forward motion.
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
            consecutive_no_path_count += 1
            pending_log_events.append("NO_PATH")

            if consecutive_no_path_count >= NO_PATH_ATTEMPTS_BEFORE_BACKTRACK:
                backtrack_start_time = current_simulation_time
                backtrack_reason = "NO_PATH"
                backtrack_count += 1
                navigation_state = "BACKTRACK"
                pending_log_events.append("BACKTRACK_STARTED")
                print(
                    "No route after repeated A* attempts. "
                    "Starting local BACKTRACK behavior."
                )
            else:
                last_path_retry_time = current_simulation_time
                navigation_state = "WAIT_FOR_PATH"
                print("No route currently available. A.M.A.S.O.N. will retry.")
        else:
            consecutive_no_path_count = 0
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
            print("Localized start cell:", starting_grid_cell)
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

    elif navigation_state == "BACKTRACK":
        # Move backward briefly to create maneuvering room.
        left_wheel_motor.setVelocity(-BACKTRACK_WHEEL_SPEED)
        right_wheel_motor.setVelocity(-BACKTRACK_WHEEL_SPEED)

        if (
            current_simulation_time - backtrack_start_time
            >= BACKTRACK_DURATION_SECONDS
        ):
            left_wheel_motor.setVelocity(0.0)
            right_wheel_motor.setVelocity(0.0)

            # Higher sensor values mean more open space.
            if left_obstacle_distance_value >= right_obstacle_distance_value:
                escape_turn_target_yaw = normalize_angle_radians(
                    robot_current_yaw + ESCAPE_TURN_ANGLE_RADIANS
                )
            else:
                escape_turn_target_yaw = normalize_angle_radians(
                    robot_current_yaw - ESCAPE_TURN_ANGLE_RADIANS
                )

            escape_turn_count += 1
            navigation_state = "ESCAPE_TURN"
            pending_log_events.append("BACKTRACK_COMPLETE")
            print(
                "Backtrack complete. Turning toward clearer side before replanning."
            )

    elif navigation_state == "ESCAPE_TURN":
        escape_heading_error = normalize_angle_radians(
            escape_turn_target_yaw - robot_current_yaw
        )

        if abs(escape_heading_error) <= WAYPOINT_HEADING_TOLERANCE:
            left_wheel_motor.setVelocity(0.0)
            right_wheel_motor.setVelocity(0.0)
            planning_reason = "LOCAL_ESCAPE"
            navigation_state = "PLAN_ROUTE"
            pending_log_events.append("ESCAPE_TURN_COMPLETE")
            print("Local escape turn complete. Replanning route.")
        elif escape_heading_error > 0.0:
            left_wheel_motor.setVelocity(-TURNING_WHEEL_SPEED)
            right_wheel_motor.setVelocity(TURNING_WHEEL_SPEED)
        else:
            left_wheel_motor.setVelocity(TURNING_WHEEL_SPEED)
            right_wheel_motor.setVelocity(-TURNING_WHEEL_SPEED)

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

        observed_map_cell_count = count_observed_map_cells(
            observed_occupancy_grid
        )

        navigation_log_writer.writerow(
            [
                round(current_simulation_time, 3),
                actual_robot_world_x_position,
                actual_robot_world_y_position,
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
                actual_robot_world_x_position,
                actual_robot_world_y_position,
                noisy_gps_world_x,
                noisy_gps_world_y,
                localized_robot_world_x_position,
                localized_robot_world_y_position,
                localized_robot_grid_cell[0],
                localized_robot_grid_cell[1],
                localization_confidence,
                observed_map_cell_count,
                backtrack_count,
                escape_turn_count,
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

        observed_map_cell_count = count_observed_map_cells(
            observed_occupancy_grid
        )

        print(
            f"Actual=({actual_robot_world_x_position:.3f},"
            f"{actual_robot_world_y_position:.3f}), "
            f"NoisyGPS=({noisy_gps_world_x:.3f},{noisy_gps_world_y:.3f}), "
            f"Localized=({localized_robot_world_x_position:.3f},"
            f"{localized_robot_world_y_position:.3f}), "
            f"LocCell={localized_robot_grid_cell}, "
            f"LocConf={localization_confidence:.3f}, "
            f"Yaw={robot_current_yaw:.3f}, "
            f"Patrol={current_patrol_location_index + 1}/{len(PATROL_LOCATIONS)}, "
            f"Waypoint={waypoint_description}, "
            f"Target={current_target_description}, "
            f"ObservedCells={observed_map_cell_count}, "
            f"TempObstacles={len(temporary_obstacles)}, "
            f"Replans={dynamic_replan_count}, "
            f"Backtracks={backtrack_count}, "
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
