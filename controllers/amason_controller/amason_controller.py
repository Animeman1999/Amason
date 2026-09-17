"""A.M.A.S.O.N. Webots Controller

Autonomous Mobile AI System for Optimized Navigation

Features:
- Differential-drive control
- GPS localization
- InertialUnit heading control
- Three distance sensors
- A* path planning
- Static obstacle mapping
- Temporary obstacle mapping
- Dynamic A* replanning
- Waypoint following
- Finite-state-machine navigation
"""

from controller import Robot
import math
import heapq


# =========================================================
# ROBOT SETUP
# =========================================================

amason_robot = Robot()

SIMULATION_TIME_STEP = int(
    amason_robot.getBasicTimeStep()
)


# =========================================================
# MOTORS
# =========================================================

left_wheel_motor = amason_robot.getDevice(
    "left wheel motor"
)

right_wheel_motor = amason_robot.getDevice(
    "right wheel motor"
)

left_wheel_motor.setPosition(float("inf"))
right_wheel_motor.setPosition(float("inf"))

left_wheel_motor.setVelocity(0.0)
right_wheel_motor.setVelocity(0.0)


# =========================================================
# GPS AND ORIENTATION
# =========================================================

gps_sensor = amason_robot.getDevice("gps")

inertial_orientation_sensor = amason_robot.getDevice(
    "inertial unit"
)

gps_sensor.enable(
    SIMULATION_TIME_STEP
)

inertial_orientation_sensor.enable(
    SIMULATION_TIME_STEP
)


# =========================================================
# DISTANCE SENSORS
# =========================================================

front_distance_sensor = amason_robot.getDevice(
    "front sensor"
)

left_front_distance_sensor = amason_robot.getDevice(
    "left front sensor"
)

right_front_distance_sensor = amason_robot.getDevice(
    "right front sensor"
)

front_distance_sensor.enable(
    SIMULATION_TIME_STEP
)

left_front_distance_sensor.enable(
    SIMULATION_TIME_STEP
)

right_front_distance_sensor.enable(
    SIMULATION_TIME_STEP
)


# =========================================================
# MOVEMENT SETTINGS
# =========================================================

FORWARD_WHEEL_SPEED = 1.0

TURNING_WHEEL_SPEED = 0.7

WAYPOINT_HEADING_TOLERANCE = math.radians(5)

WAYPOINT_DISTANCE_TOLERANCE = 0.10


# =========================================================
# DYNAMIC OBSTACLE SETTINGS
# =========================================================

# A new obstacle is initially detected below this value.
#
# With the current sensor lookup table:
#
# 0 meters = 0
# 1 meter  = 1000
#
# Therefore 300 is approximately 0.30 meters.
OBSTACLE_DETECTION_THRESHOLD = 300


# Once confirmation begins, the obstacle must become
# farther away than this value before being considered
# cleared.
#
# This provides hysteresis and prevents readings near
# 299-300 from repeatedly switching between detected
# and clear.
OBSTACLE_CLEAR_THRESHOLD = 350


# An unexpected obstacle must remain visible this long
# before it is added to the temporary map.
OBSTACLE_CONFIRMATION_TIME_SECONDS = 0.50


# Temporary obstacles are forgotten after this amount of
# time unless the robot sees the same obstacle again.
TEMPORARY_OBSTACLE_LIFETIME_SECONDS = 30.0


# A detected obstacle blocks its grid cell plus this many
# neighboring cells in each direction.
#
# 1 creates a 3 x 3 blocked region.
TEMPORARY_OBSTACLE_INFLATION_CELLS = 1


# Sensor position estimates can change slightly as the
# robot moves. Treat detections within this many cells as
# being the same temporary obstacle.
TEMPORARY_OBSTACLE_MATCH_RADIUS_CELLS = 1


# If no path can currently be found, wait this long before
# trying A* again.
PATH_RETRY_INTERVAL_SECONDS = 1.0


# =========================================================
# SENSOR MOUNTING LOCATIONS
# =========================================================
#
# Robot coordinate system:
#
# +X = robot forward
# +Y = robot left
#
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
# ARENA / GRID SETTINGS
# =========================================================

ARENA_SIZE_METERS = 5.0

GRID_CELL_SIZE_METERS = 0.25

GRID_CELL_COUNT = int(
    ARENA_SIZE_METERS
    / GRID_CELL_SIZE_METERS
)

ARENA_MINIMUM_COORDINATE = (
    -ARENA_SIZE_METERS / 2.0
)


# =========================================================
# STATIC OBSTACLE SETTINGS
# =========================================================

BOX_ONE_CENTER_X = 1.20
BOX_ONE_CENTER_Y = 0.40

BOX_ONE_SIZE_X = 0.20
BOX_ONE_SIZE_Y = 0.40


BOX_TWO_CENTER_X = 0.70
BOX_TWO_CENTER_Y = 1.10

BOX_TWO_SIZE_X = 0.20
BOX_TWO_SIZE_Y = 0.40


# Extra clearance around known obstacles.
STATIC_OBSTACLE_SAFETY_MARGIN = 0.30


# =========================================================
# START AND GOAL
# =========================================================

ROBOT_START_WORLD_POSITION = (
    0.0,
    0.0
)

ROBOT_GOAL_WORLD_POSITION = (
    1.8,
    1.20
)


# =========================================================
# ANGLE FUNCTIONS
# =========================================================

def normalize_angle_radians(
    angle_radians
):
    """
    Normalize an angle to the range
    -pi through +pi.
    """

    while angle_radians > math.pi:

        angle_radians -= (
            2 * math.pi
        )

    while angle_radians < -math.pi:

        angle_radians += (
            2 * math.pi
        )

    return angle_radians


# =========================================================
# GRID / WORLD CONVERSION
# =========================================================

def convert_world_position_to_grid_cell(
    world_x_position,
    world_y_position
):
    """
    Convert Webots world coordinates into
    occupancy-grid coordinates.
    """

    grid_column = int(
        (
            world_x_position
            - ARENA_MINIMUM_COORDINATE
        )
        / GRID_CELL_SIZE_METERS
    )

    grid_row = int(
        (
            world_y_position
            - ARENA_MINIMUM_COORDINATE
        )
        / GRID_CELL_SIZE_METERS
    )

    grid_column = max(
        0,
        min(
            GRID_CELL_COUNT - 1,
            grid_column
        )
    )

    grid_row = max(
        0,
        min(
            GRID_CELL_COUNT - 1,
            grid_row
        )
    )

    return (
        grid_column,
        grid_row
    )


def convert_grid_cell_to_world_position(
    grid_column,
    grid_row
):
    """
    Convert a grid cell into the Webots
    world coordinates at the center of
    the cell.
    """

    world_x_position = (
        ARENA_MINIMUM_COORDINATE
        + (
            grid_column + 0.5
        )
        * GRID_CELL_SIZE_METERS
    )

    world_y_position = (
        ARENA_MINIMUM_COORDINATE
        + (
            grid_row + 0.5
        )
        * GRID_CELL_SIZE_METERS
    )

    return (
        world_x_position,
        world_y_position
    )


# =========================================================
# OCCUPANCY GRID FUNCTIONS
# =========================================================

def create_empty_occupancy_grid():
    """
    Create an empty occupancy grid.

    0 = open cell
    1 = blocked cell
    """

    return [
        [
            0
            for grid_row
            in range(
                GRID_CELL_COUNT
            )
        ]
        for grid_column
        in range(
            GRID_CELL_COUNT
        )
    ]


def copy_occupancy_grid(
    source_grid
):
    """
    Create an independent copy of an
    occupancy grid.
    """

    return [
        grid_column.copy()
        for grid_column
        in source_grid
    ]


def mark_rectangular_obstacle_on_grid(
    occupancy_grid,
    obstacle_center_x,
    obstacle_center_y,
    obstacle_size_x,
    obstacle_size_y,
    safety_margin
):
    """
    Mark a known rectangular obstacle
    on the occupancy grid.
    """

    obstacle_minimum_x = (
        obstacle_center_x
        - obstacle_size_x / 2.0
        - safety_margin
    )

    obstacle_maximum_x = (
        obstacle_center_x
        + obstacle_size_x / 2.0
        + safety_margin
    )

    obstacle_minimum_y = (
        obstacle_center_y
        - obstacle_size_y / 2.0
        - safety_margin
    )

    obstacle_maximum_y = (
        obstacle_center_y
        + obstacle_size_y / 2.0
        + safety_margin
    )

    (
        minimum_grid_column,
        minimum_grid_row
    ) = (
        convert_world_position_to_grid_cell(
            obstacle_minimum_x,
            obstacle_minimum_y
        )
    )

    (
        maximum_grid_column,
        maximum_grid_row
    ) = (
        convert_world_position_to_grid_cell(
            obstacle_maximum_x,
            obstacle_maximum_y
        )
    )

    for obstacle_grid_column in range(
        minimum_grid_column,
        maximum_grid_column + 1
    ):

        for obstacle_grid_row in range(
            minimum_grid_row,
            maximum_grid_row + 1
        ):

            occupancy_grid[
                obstacle_grid_column
            ][
                obstacle_grid_row
            ] = 1


def mark_temporary_obstacle_on_grid(
    occupancy_grid,
    obstacle_grid_cell
):
    """
    Mark a temporary obstacle and its
    surrounding safety cells as blocked.
    """

    obstacle_grid_column = (
        obstacle_grid_cell[0]
    )

    obstacle_grid_row = (
        obstacle_grid_cell[1]
    )

    for column_offset in range(
        -TEMPORARY_OBSTACLE_INFLATION_CELLS,
        TEMPORARY_OBSTACLE_INFLATION_CELLS + 1
    ):

        for row_offset in range(
            -TEMPORARY_OBSTACLE_INFLATION_CELLS,
            TEMPORARY_OBSTACLE_INFLATION_CELLS + 1
        ):

            blocked_grid_column = (
                obstacle_grid_column
                + column_offset
            )

            blocked_grid_row = (
                obstacle_grid_row
                + row_offset
            )

            if (
                0
                <= blocked_grid_column
                < GRID_CELL_COUNT
                and
                0
                <= blocked_grid_row
                < GRID_CELL_COUNT
            ):

                occupancy_grid[
                    blocked_grid_column
                ][
                    blocked_grid_row
                ] = 1


# =========================================================
# TEMPORARY OBSTACLE MEMORY
# =========================================================
#
# Dictionary format:
#
# {
#     (grid_column, grid_row):
#         last_detection_time
# }
#
# =========================================================

temporary_obstacles = {}


def remove_expired_temporary_obstacles(
    current_simulation_time
):
    """
    Remove temporary obstacles that have
    not been detected recently.
    """

    expired_obstacle_cells = []

    for (
        obstacle_grid_cell,
        last_detection_time
    ) in temporary_obstacles.items():

        obstacle_age_seconds = (
            current_simulation_time
            - last_detection_time
        )

        if (
            obstacle_age_seconds
            >
            TEMPORARY_OBSTACLE_LIFETIME_SECONDS
        ):

            expired_obstacle_cells.append(
                obstacle_grid_cell
            )

    for obstacle_grid_cell in (
        expired_obstacle_cells
    ):

        del temporary_obstacles[
            obstacle_grid_cell
        ]

        print(
            "Temporary obstacle expired:",
            obstacle_grid_cell
        )


def find_matching_temporary_obstacle(
    detected_grid_cell
):
    """
    Determine whether a newly estimated
    obstacle position corresponds to an
    obstacle already stored in memory.
    """

    detected_grid_column = (
        detected_grid_cell[0]
    )

    detected_grid_row = (
        detected_grid_cell[1]
    )

    for existing_grid_cell in (
        temporary_obstacles.keys()
    ):

        existing_grid_column = (
            existing_grid_cell[0]
        )

        existing_grid_row = (
            existing_grid_cell[1]
        )

        column_difference = abs(
            existing_grid_column
            - detected_grid_column
        )

        row_difference = abs(
            existing_grid_row
            - detected_grid_row
        )

        if (
            column_difference
            <=
            TEMPORARY_OBSTACLE_MATCH_RADIUS_CELLS
            and
            row_difference
            <=
            TEMPORARY_OBSTACLE_MATCH_RADIUS_CELLS
        ):

            return existing_grid_cell

    return None


def add_or_refresh_temporary_obstacle(
    detected_grid_cell,
    current_simulation_time
):
    """
    Add a newly discovered obstacle or
    refresh an existing nearby obstacle.
    """

    matching_obstacle_cell = (
        find_matching_temporary_obstacle(
            detected_grid_cell
        )
    )

    if matching_obstacle_cell is not None:

        temporary_obstacles[
            matching_obstacle_cell
        ] = current_simulation_time

        return matching_obstacle_cell

    temporary_obstacles[
        detected_grid_cell
    ] = current_simulation_time

    return detected_grid_cell


def build_current_occupancy_grid(
    current_simulation_time
):
    """
    Build the current planning map using:

    - known static obstacles
    - remembered temporary obstacles
    """

    remove_expired_temporary_obstacles(
        current_simulation_time
    )

    combined_occupancy_grid = (
        copy_occupancy_grid(
            static_occupancy_grid
        )
    )

    for obstacle_grid_cell in (
        temporary_obstacles.keys()
    ):

        mark_temporary_obstacle_on_grid(
            combined_occupancy_grid,
            obstacle_grid_cell
        )

    return combined_occupancy_grid


def grid_cell_is_static_obstacle(
    grid_cell
):
    """
    Determine whether a detected grid cell
    is already part of the known static map.
    """

    grid_column = grid_cell[0]
    grid_row = grid_cell[1]

    return (
        static_occupancy_grid[
            grid_column
        ][
            grid_row
        ]
        == 1
    )


# =========================================================
# A* FUNCTIONS
# =========================================================

def calculate_manhattan_distance(
    first_grid_cell,
    second_grid_cell
):
    """
    Calculate Manhattan distance between
    two grid cells.
    """

    column_distance = abs(
        first_grid_cell[0]
        - second_grid_cell[0]
    )

    row_distance = abs(
        first_grid_cell[1]
        - second_grid_cell[1]
    )

    return (
        column_distance
        + row_distance
    )


def calculate_astar_path(
    occupancy_grid,
    starting_grid_cell,
    goal_grid_cell
):
    """
    Calculate an optimal grid path using A*.

    Movement is limited to four directions:

    - left
    - right
    - forward grid row
    - backward grid row
    """

    frontier_priority_queue = []

    heapq.heappush(
        frontier_priority_queue,
        (
            0,
            starting_grid_cell
        )
    )

    previous_grid_cell = {}

    movement_cost_from_start = {
        starting_grid_cell: 0
    }

    while frontier_priority_queue:

        (
            current_estimated_total_cost,
            current_grid_cell
        ) = heapq.heappop(
            frontier_priority_queue
        )

        # -------------------------------------------------
        # GOAL FOUND
        # -------------------------------------------------

        if (
            current_grid_cell
            == goal_grid_cell
        ):

            calculated_path = [
                current_grid_cell
            ]

            while (
                current_grid_cell
                in previous_grid_cell
            ):

                current_grid_cell = (
                    previous_grid_cell[
                        current_grid_cell
                    ]
                )

                calculated_path.append(
                    current_grid_cell
                )

            calculated_path.reverse()

            return calculated_path


        (
            current_grid_column,
            current_grid_row
        ) = current_grid_cell


        neighboring_grid_cells = [

            (
                current_grid_column + 1,
                current_grid_row
            ),

            (
                current_grid_column - 1,
                current_grid_row
            ),

            (
                current_grid_column,
                current_grid_row + 1
            ),

            (
                current_grid_column,
                current_grid_row - 1
            )
        ]


        for neighboring_grid_cell in (
            neighboring_grid_cells
        ):

            (
                neighboring_grid_column,
                neighboring_grid_row
            ) = neighboring_grid_cell


            # ---------------------------------------------
            # GRID BOUNDARY CHECK
            # ---------------------------------------------

            if (
                neighboring_grid_column < 0
                or
                neighboring_grid_column
                >= GRID_CELL_COUNT
            ):

                continue

            if (
                neighboring_grid_row < 0
                or
                neighboring_grid_row
                >= GRID_CELL_COUNT
            ):

                continue


            # ---------------------------------------------
            # OBSTACLE CHECK
            # ---------------------------------------------

            if (
                occupancy_grid[
                    neighboring_grid_column
                ][
                    neighboring_grid_row
                ]
                == 1
            ):

                continue


            new_movement_cost_from_start = (
                movement_cost_from_start[
                    current_grid_cell
                ]
                + 1
            )


            if (
                neighboring_grid_cell
                not in movement_cost_from_start
                or
                new_movement_cost_from_start
                <
                movement_cost_from_start[
                    neighboring_grid_cell
                ]
            ):

                previous_grid_cell[
                    neighboring_grid_cell
                ] = current_grid_cell

                movement_cost_from_start[
                    neighboring_grid_cell
                ] = (
                    new_movement_cost_from_start
                )

                estimated_total_path_cost = (
                    new_movement_cost_from_start
                    +
                    calculate_manhattan_distance(
                        neighboring_grid_cell,
                        goal_grid_cell
                    )
                )

                heapq.heappush(
                    frontier_priority_queue,
                    (
                        estimated_total_path_cost,
                        neighboring_grid_cell
                    )
                )

    return None


# =========================================================
# WAYPOINT CREATION
# =========================================================

def create_navigation_waypoints(
    planned_path
):
    """
    Convert an A* grid path into Webots
    world-space navigation waypoints.
    """

    new_navigation_waypoints = []

    if planned_path is None:

        return (
            new_navigation_waypoints
        )

    # Skip path[0] because it is the cell
    # containing the robot's current location.
    for path_grid_cell in (
        planned_path[1:]
    ):

        world_space_waypoint = (
            convert_grid_cell_to_world_position(
                path_grid_cell[0],
                path_grid_cell[1]
            )
        )

        new_navigation_waypoints.append(
            world_space_waypoint
        )

    # Use the exact requested goal coordinate
    # instead of the center of the final cell.
    if (
        len(
            new_navigation_waypoints
        )
        > 0
    ):

        new_navigation_waypoints[-1] = (
            ROBOT_GOAL_WORLD_POSITION
        )

    return (
        new_navigation_waypoints
    )


# =========================================================
# SENSOR FUNCTIONS
# =========================================================

def convert_sensor_value_to_meters(
    sensor_value
):
    """
    Convert the current 0-1000 sensor value
    into approximate meters.
    """

    return (
        sensor_value / 1000.0
    )


def find_closest_detected_obstacle(
    front_sensor_value,
    left_sensor_value,
    right_sensor_value,
    detection_threshold
):
    """
    Return information about the closest
    detected obstacle below the supplied
    threshold.
    """

    possible_detections = []


    if (
        front_sensor_value
        < detection_threshold
    ):

        possible_detections.append(
            {
                "sensor_name": "FRONT",
                "sensor_value":
                    front_sensor_value,
                "local_x":
                    FRONT_SENSOR_LOCAL_X,
                "local_y":
                    FRONT_SENSOR_LOCAL_Y,
                "sensor_angle":
                    FRONT_SENSOR_ANGLE
            }
        )


    if (
        left_sensor_value
        < detection_threshold
    ):

        possible_detections.append(
            {
                "sensor_name": "LEFT",
                "sensor_value":
                    left_sensor_value,
                "local_x":
                    LEFT_SENSOR_LOCAL_X,
                "local_y":
                    LEFT_SENSOR_LOCAL_Y,
                "sensor_angle":
                    LEFT_SENSOR_ANGLE
            }
        )


    if (
        right_sensor_value
        < detection_threshold
    ):

        possible_detections.append(
            {
                "sensor_name": "RIGHT",
                "sensor_value":
                    right_sensor_value,
                "local_x":
                    RIGHT_SENSOR_LOCAL_X,
                "local_y":
                    RIGHT_SENSOR_LOCAL_Y,
                "sensor_angle":
                    RIGHT_SENSOR_ANGLE
            }
        )


    if (
        len(
            possible_detections
        )
        == 0
    ):

        return None


    return min(
        possible_detections,
        key=lambda detection:
            detection[
                "sensor_value"
            ]
    )


def estimate_obstacle_world_position(
    robot_world_x_position,
    robot_world_y_position,
    robot_current_yaw,
    obstacle_detection
):
    """
    Estimate obstacle world position using:

    - GPS robot position
    - robot yaw
    - sensor mounting position
    - sensor mounting angle
    - measured distance
    """

    sensor_local_x = (
        obstacle_detection[
            "local_x"
        ]
    )

    sensor_local_y = (
        obstacle_detection[
            "local_y"
        ]
    )

    obstacle_distance_meters = (
        convert_sensor_value_to_meters(
            obstacle_detection[
                "sensor_value"
            ]
        )
    )


    # -----------------------------------------------------
    # SENSOR WORLD POSITION
    # -----------------------------------------------------

    sensor_world_x = (
        robot_world_x_position
        +
        sensor_local_x
        * math.cos(
            robot_current_yaw
        )
        -
        sensor_local_y
        * math.sin(
            robot_current_yaw
        )
    )

    sensor_world_y = (
        robot_world_y_position
        +
        sensor_local_x
        * math.sin(
            robot_current_yaw
        )
        +
        sensor_local_y
        * math.cos(
            robot_current_yaw
        )
    )


    # -----------------------------------------------------
    # SENSOR WORLD HEADING
    # -----------------------------------------------------

    sensor_world_heading = (
        robot_current_yaw
        +
        obstacle_detection[
            "sensor_angle"
        ]
    )


    # -----------------------------------------------------
    # ESTIMATED OBSTACLE LOCATION
    # -----------------------------------------------------

    estimated_obstacle_world_x = (
        sensor_world_x
        +
        obstacle_distance_meters
        * math.cos(
            sensor_world_heading
        )
    )

    estimated_obstacle_world_y = (
        sensor_world_y
        +
        obstacle_distance_meters
        * math.sin(
            sensor_world_heading
        )
    )


    return (
        estimated_obstacle_world_x,
        estimated_obstacle_world_y
    )


# =========================================================
# CREATE STATIC OCCUPANCY GRID
# =========================================================

static_occupancy_grid = (
    create_empty_occupancy_grid()
)


# =========================================================
# ADD KNOWN BOX ONE
# =========================================================

mark_rectangular_obstacle_on_grid(
    occupancy_grid=
        static_occupancy_grid,

    obstacle_center_x=
        BOX_ONE_CENTER_X,

    obstacle_center_y=
        BOX_ONE_CENTER_Y,

    obstacle_size_x=
        BOX_ONE_SIZE_X,

    obstacle_size_y=
        BOX_ONE_SIZE_Y,

    safety_margin=
        STATIC_OBSTACLE_SAFETY_MARGIN
)


# =========================================================
# ADD KNOWN BOX TWO
# =========================================================

mark_rectangular_obstacle_on_grid(
    occupancy_grid=
        static_occupancy_grid,

    obstacle_center_x=
        BOX_TWO_CENTER_X,

    obstacle_center_y=
        BOX_TWO_CENTER_Y,

    obstacle_size_x=
        BOX_TWO_SIZE_X,

    obstacle_size_y=
        BOX_TWO_SIZE_Y,

    safety_margin=
        STATIC_OBSTACLE_SAFETY_MARGIN
)


# =========================================================
# INITIAL A* PATH
# =========================================================

starting_grid_cell = (
    convert_world_position_to_grid_cell(
        ROBOT_START_WORLD_POSITION[0],
        ROBOT_START_WORLD_POSITION[1]
    )
)

goal_grid_cell = (
    convert_world_position_to_grid_cell(
        ROBOT_GOAL_WORLD_POSITION[0],
        ROBOT_GOAL_WORLD_POSITION[1]
    )
)


current_occupancy_grid = (
    build_current_occupancy_grid(
        amason_robot.getTime()
    )
)


planned_grid_path = (
    calculate_astar_path(
        current_occupancy_grid,
        starting_grid_cell,
        goal_grid_cell
    )
)


navigation_waypoints = (
    create_navigation_waypoints(
        planned_grid_path
    )
)


# =========================================================
# PRINT INITIAL ROUTE
# =========================================================

print()

print(
    "=========================================="
)

print(
    "A.M.A.S.O.N. AUTONOMOUS NAVIGATION"
)

print(
    "=========================================="
)

print(
    "Start location:",
    ROBOT_START_WORLD_POSITION
)

print(
    "Goal location:",
    ROBOT_GOAL_WORLD_POSITION
)


if planned_grid_path is None:

    print(
        "ERROR: A* COULD NOT FIND AN INITIAL PATH"
    )

else:

    print(
        "Initial A* path found."
    )

    print(
        "Grid cells:",
        len(
            planned_grid_path
        )
    )

    print(
        "Navigation waypoints:",
        len(
            navigation_waypoints
        )
    )

    for (
        waypoint_number,
        waypoint_world_position
    ) in enumerate(
        navigation_waypoints,
        start=1
    ):

        print(
            f"Waypoint {waypoint_number}: "
            f"X={waypoint_world_position[0]:.3f}, "
            f"Y={waypoint_world_position[1]:.3f}"
        )


print(
    "=========================================="
)

print()


# =========================================================
# NAVIGATION FSM INITIALIZATION
# =========================================================

if (
    planned_grid_path is None
    or
    len(
        navigation_waypoints
    )
    == 0
):

    navigation_state = (
        "STOPPED"
    )

else:

    navigation_state = (
        "TURN_TO_WAYPOINT"
    )


current_waypoint_index = 0

obstacle_confirmation_start_time = 0.0

last_path_retry_time = 0.0


# =========================================================
# MAIN CONTROL LOOP
# =========================================================

while (
    amason_robot.step(
        SIMULATION_TIME_STEP
    )
    != -1
):

    current_simulation_time = (
        amason_robot.getTime()
    )


    # -----------------------------------------------------
    # CURRENT POSITION
    # -----------------------------------------------------

    current_gps_position = (
        gps_sensor.getValues()
    )

    robot_world_x_position = (
        current_gps_position[0]
    )

    robot_world_y_position = (
        current_gps_position[1]
    )


    # -----------------------------------------------------
    # CURRENT ORIENTATION
    # -----------------------------------------------------

    current_roll_pitch_yaw = (
        inertial_orientation_sensor
        .getRollPitchYaw()
    )

    robot_current_yaw = (
        current_roll_pitch_yaw[2]
    )


    # -----------------------------------------------------
    # DISTANCE SENSOR VALUES
    # -----------------------------------------------------

    front_obstacle_distance_value = (
        front_distance_sensor.getValue()
    )

    left_obstacle_distance_value = (
        left_front_distance_sensor.getValue()
    )

    right_obstacle_distance_value = (
        right_front_distance_sensor.getValue()
    )


    # =====================================================
    # FIND CLOSEST DETECTED OBSTACLE
    # =====================================================

    closest_obstacle_detection = (
        find_closest_detected_obstacle(
            front_obstacle_distance_value,
            left_obstacle_distance_value,
            right_obstacle_distance_value,
            OBSTACLE_DETECTION_THRESHOLD
        )
    )


    detected_obstacle_grid_cell = None

    matching_temporary_obstacle_cell = None

    detection_is_known_static_obstacle = False


    # =====================================================
    # ESTIMATE OBSTACLE POSITION
    # =====================================================

    if (
        closest_obstacle_detection
        is not None
    ):

        (
            estimated_obstacle_world_x,
            estimated_obstacle_world_y
        ) = (
            estimate_obstacle_world_position(
                robot_world_x_position,
                robot_world_y_position,
                robot_current_yaw,
                closest_obstacle_detection
            )
        )


        detected_obstacle_grid_cell = (
            convert_world_position_to_grid_cell(
                estimated_obstacle_world_x,
                estimated_obstacle_world_y
            )
        )


        # -------------------------------------------------
        # IS IT ALREADY A KNOWN STATIC OBSTACLE?
        # -------------------------------------------------

        detection_is_known_static_obstacle = (
            grid_cell_is_static_obstacle(
                detected_obstacle_grid_cell
            )
        )


        # -------------------------------------------------
        # IS IT ALREADY A TEMPORARY OBSTACLE?
        # -------------------------------------------------

        matching_temporary_obstacle_cell = (
            find_matching_temporary_obstacle(
                detected_obstacle_grid_cell
            )
        )


        # -------------------------------------------------
        # REFRESH KNOWN TEMPORARY OBSTACLE
        # -------------------------------------------------
        #
        # Seeing the same temporary obstacle again should
        # NOT trigger another A* replan.
        # -------------------------------------------------

        if (
            matching_temporary_obstacle_cell
            is not None
        ):

            temporary_obstacles[
                matching_temporary_obstacle_cell
            ] = current_simulation_time


    # =====================================================
    # COMPLETED NAVIGATION CHECK
    # =====================================================

    if (
        len(
            navigation_waypoints
        )
        > 0
        and
        current_waypoint_index
        >=
        len(
            navigation_waypoints
        )
    ):

        navigation_state = (
            "GOAL_REACHED"
        )


    # =====================================================
    # NEW / UNKNOWN OBSTACLE DETECTED
    # =====================================================
    #
    # Only an obstacle that is:
    #
    # 1. not part of the static map
    # 2. not already in the temporary map
    #
    # begins the confirmation process.
    # =====================================================

    if (
        navigation_state
        == "DRIVE_TO_WAYPOINT"
        and
        closest_obstacle_detection
        is not None
        and
        not detection_is_known_static_obstacle
        and
        matching_temporary_obstacle_cell
        is None
    ):

        left_wheel_motor.setVelocity(
            0.0
        )

        right_wheel_motor.setVelocity(
            0.0
        )

        obstacle_confirmation_start_time = (
            current_simulation_time
        )

        navigation_state = (
            "CONFIRM_OBSTACLE"
        )

        print()

        print(
            "Unexpected obstacle detected by",
            closest_obstacle_detection[
                "sensor_name"
            ],
            "sensor."
        )

        print(
            "Estimated grid cell:",
            detected_obstacle_grid_cell
        )


    # =====================================================
    # CONFIRM OBSTACLE
    # =====================================================

    if (
        navigation_state
        == "CONFIRM_OBSTACLE"
    ):

        left_wheel_motor.setVelocity(
            0.0
        )

        right_wheel_motor.setVelocity(
            0.0
        )


        # Use the larger clear threshold while
        # confirmation is in progress.
        confirmation_obstacle_detection = (
            find_closest_detected_obstacle(
                front_obstacle_distance_value,
                left_obstacle_distance_value,
                right_obstacle_distance_value,
                OBSTACLE_CLEAR_THRESHOLD
            )
        )


        # -------------------------------------------------
        # OBSTACLE DISAPPEARED
        # -------------------------------------------------

        if (
            confirmation_obstacle_detection
            is None
        ):

            print(
                "Obstacle cleared before confirmation."
            )

            navigation_state = (
                "TURN_TO_WAYPOINT"
            )


        else:

            obstacle_visible_time = (
                current_simulation_time
                -
                obstacle_confirmation_start_time
            )


            # ---------------------------------------------
            # OBSTACLE CONFIRMED
            # ---------------------------------------------

            if (
                obstacle_visible_time
                >=
                OBSTACLE_CONFIRMATION_TIME_SECONDS
            ):

                (
                    estimated_obstacle_world_x,
                    estimated_obstacle_world_y
                ) = (
                    estimate_obstacle_world_position(
                        robot_world_x_position,
                        robot_world_y_position,
                        robot_current_yaw,
                        confirmation_obstacle_detection
                    )
                )


                detected_obstacle_grid_cell = (
                    convert_world_position_to_grid_cell(
                        estimated_obstacle_world_x,
                        estimated_obstacle_world_y
                    )
                )


                # -----------------------------------------
                # CHECK WHETHER STATIC MAP ALREADY KNOWS IT
                # -----------------------------------------

                if (
                    grid_cell_is_static_obstacle(
                        detected_obstacle_grid_cell
                    )
                ):

                    print(
                        "Detected obstacle is already "
                        "part of the static map."
                    )

                    navigation_state = (
                        "TURN_TO_WAYPOINT"
                    )


                else:

                    # -------------------------------------
                    # CHECK WHETHER TEMPORARY MAP KNOWS IT
                    # -------------------------------------

                    matching_temporary_obstacle_cell = (
                        find_matching_temporary_obstacle(
                            detected_obstacle_grid_cell
                        )
                    )


                    if (
                        matching_temporary_obstacle_cell
                        is not None
                    ):

                        temporary_obstacles[
                            matching_temporary_obstacle_cell
                        ] = (
                            current_simulation_time
                        )

                        print(
                            "Detected obstacle is already "
                            "in temporary map:",
                            matching_temporary_obstacle_cell
                        )

                        navigation_state = (
                            "TURN_TO_WAYPOINT"
                        )


                    else:

                        # ---------------------------------
                        # GENUINELY NEW OBSTACLE
                        # ---------------------------------

                        stored_obstacle_grid_cell = (
                            add_or_refresh_temporary_obstacle(
                                detected_obstacle_grid_cell,
                                current_simulation_time
                            )
                        )


                        print()

                        print(
                            "=========================================="
                        )

                        print(
                            "NEW TEMPORARY OBSTACLE ADDED"
                        )

                        print(
                            "Sensor:",
                            confirmation_obstacle_detection[
                                "sensor_name"
                            ]
                        )

                        print(
                            "Estimated world position:",
                            (
                                round(
                                    estimated_obstacle_world_x,
                                    3
                                ),
                                round(
                                    estimated_obstacle_world_y,
                                    3
                                )
                            )
                        )

                        print(
                            "Grid cell:",
                            stored_obstacle_grid_cell
                        )

                        print(
                            "Replanning route..."
                        )

                        print(
                            "=========================================="
                        )

                        print()

                        navigation_state = (
                            "REPLAN"
                        )


    # =====================================================
    # REPLAN A* PATH
    # =====================================================

    if navigation_state == "REPLAN":

        left_wheel_motor.setVelocity(
            0.0
        )

        right_wheel_motor.setVelocity(
            0.0
        )


        current_robot_grid_cell = (
            convert_world_position_to_grid_cell(
                robot_world_x_position,
                robot_world_y_position
            )
        )


        current_occupancy_grid = (
            build_current_occupancy_grid(
                current_simulation_time
            )
        )


        # The robot must always be able to
        # start from its current grid cell.
        current_occupancy_grid[
            current_robot_grid_cell[0]
        ][
            current_robot_grid_cell[1]
        ] = 0


        replanned_grid_path = (
            calculate_astar_path(
                current_occupancy_grid,
                current_robot_grid_cell,
                goal_grid_cell
            )
        )


        # -------------------------------------------------
        # NO PATH CURRENTLY AVAILABLE
        # -------------------------------------------------

        if (
            replanned_grid_path
            is None
        ):

            print(
                "No route currently available."
            )

            print(
                "A.M.A.S.O.N. will wait "
                "and try again."
            )

            navigation_waypoints = []

            planned_grid_path = None

            last_path_retry_time = (
                current_simulation_time
            )

            navigation_state = (
                "WAIT_FOR_PATH"
            )


        # -------------------------------------------------
        # NEW PATH FOUND
        # -------------------------------------------------

        else:

            planned_grid_path = (
                replanned_grid_path
            )

            navigation_waypoints = (
                create_navigation_waypoints(
                    planned_grid_path
                )
            )

            current_waypoint_index = 0


            print()

            print(
                "=========================================="
            )

            print(
                "NEW A* PATH FOUND"
            )

            print(
                "Starting cell:",
                current_robot_grid_cell
            )

            print(
                "Grid cells:",
                len(
                    planned_grid_path
                )
            )

            print(
                "New waypoints:",
                len(
                    navigation_waypoints
                )
            )


            for (
                waypoint_number,
                waypoint_world_position
            ) in enumerate(
                navigation_waypoints,
                start=1
            ):

                print(
                    f"Waypoint {waypoint_number}: "
                    f"X={waypoint_world_position[0]:.3f}, "
                    f"Y={waypoint_world_position[1]:.3f}"
                )


            print(
                "=========================================="
            )

            print()


            if (
                len(
                    navigation_waypoints
                )
                == 0
            ):

                navigation_state = (
                    "GOAL_REACHED"
                )

            else:

                navigation_state = (
                    "TURN_TO_WAYPOINT"
                )


    # =====================================================
    # WAIT FOR PATH
    # =====================================================

    elif (
        navigation_state
        == "WAIT_FOR_PATH"
    ):

        left_wheel_motor.setVelocity(
            0.0
        )

        right_wheel_motor.setVelocity(
            0.0
        )


        time_since_last_retry = (
            current_simulation_time
            -
            last_path_retry_time
        )


        if (
            time_since_last_retry
            >=
            PATH_RETRY_INTERVAL_SECONDS
        ):

            last_path_retry_time = (
                current_simulation_time
            )

            navigation_state = (
                "REPLAN"
            )


    # =====================================================
    # TURN TOWARD WAYPOINT
    # =====================================================

    elif (
        navigation_state
        == "TURN_TO_WAYPOINT"
    ):

        (
            waypoint_target_x_position,
            waypoint_target_y_position
        ) = navigation_waypoints[
            current_waypoint_index
        ]


        distance_to_waypoint_x = (
            waypoint_target_x_position
            -
            robot_world_x_position
        )

        distance_to_waypoint_y = (
            waypoint_target_y_position
            -
            robot_world_y_position
        )


        straight_line_distance_to_waypoint = (
            math.sqrt(
                distance_to_waypoint_x ** 2
                +
                distance_to_waypoint_y ** 2
            )
        )


        # -------------------------------------------------
        # WAYPOINT ALREADY REACHED
        # -------------------------------------------------

        if (
            straight_line_distance_to_waypoint
            <=
            WAYPOINT_DISTANCE_TOLERANCE
        ):

            current_waypoint_index += 1


            if (
                current_waypoint_index
                >=
                len(
                    navigation_waypoints
                )
            ):

                navigation_state = (
                    "GOAL_REACHED"
                )

            else:

                navigation_state = (
                    "TURN_TO_WAYPOINT"
                )


        else:

            desired_waypoint_heading = (
                math.atan2(
                    distance_to_waypoint_y,
                    distance_to_waypoint_x
                )
            )


            waypoint_heading_error = (
                normalize_angle_radians(
                    desired_waypoint_heading
                    -
                    robot_current_yaw
                )
            )


            # ---------------------------------------------
            # HEADING CORRECT
            # ---------------------------------------------

            if (
                abs(
                    waypoint_heading_error
                )
                <=
                WAYPOINT_HEADING_TOLERANCE
            ):

                left_wheel_motor.setVelocity(
                    0.0
                )

                right_wheel_motor.setVelocity(
                    0.0
                )

                navigation_state = (
                    "DRIVE_TO_WAYPOINT"
                )


            # ---------------------------------------------
            # TARGET IS TO LEFT
            # ---------------------------------------------

            elif (
                waypoint_heading_error
                > 0
            ):

                left_wheel_motor.setVelocity(
                    -TURNING_WHEEL_SPEED
                )

                right_wheel_motor.setVelocity(
                    TURNING_WHEEL_SPEED
                )


            # ---------------------------------------------
            # TARGET IS TO RIGHT
            # ---------------------------------------------

            else:

                left_wheel_motor.setVelocity(
                    TURNING_WHEEL_SPEED
                )

                right_wheel_motor.setVelocity(
                    -TURNING_WHEEL_SPEED
                )


    # =====================================================
    # DRIVE TOWARD WAYPOINT
    # =====================================================

    elif (
        navigation_state
        == "DRIVE_TO_WAYPOINT"
    ):

        (
            waypoint_target_x_position,
            waypoint_target_y_position
        ) = navigation_waypoints[
            current_waypoint_index
        ]


        distance_to_waypoint_x = (
            waypoint_target_x_position
            -
            robot_world_x_position
        )

        distance_to_waypoint_y = (
            waypoint_target_y_position
            -
            robot_world_y_position
        )


        straight_line_distance_to_waypoint = (
            math.sqrt(
                distance_to_waypoint_x ** 2
                +
                distance_to_waypoint_y ** 2
            )
        )


        desired_waypoint_heading = (
            math.atan2(
                distance_to_waypoint_y,
                distance_to_waypoint_x
            )
        )


        waypoint_heading_error = (
            normalize_angle_radians(
                desired_waypoint_heading
                -
                robot_current_yaw
            )
        )


        # -------------------------------------------------
        # WAYPOINT REACHED
        # -------------------------------------------------

        if (
            straight_line_distance_to_waypoint
            <=
            WAYPOINT_DISTANCE_TOLERANCE
        ):

            current_waypoint_index += 1


            left_wheel_motor.setVelocity(
                0.0
            )

            right_wheel_motor.setVelocity(
                0.0
            )


            if (
                current_waypoint_index
                >=
                len(
                    navigation_waypoints
                )
            ):

                navigation_state = (
                    "GOAL_REACHED"
                )

            else:

                navigation_state = (
                    "TURN_TO_WAYPOINT"
                )


        # -------------------------------------------------
        # HEADING NEEDS CORRECTION
        # -------------------------------------------------

        elif (
            abs(
                waypoint_heading_error
            )
            >
            WAYPOINT_HEADING_TOLERANCE
        ):

            left_wheel_motor.setVelocity(
                0.0
            )

            right_wheel_motor.setVelocity(
                0.0
            )

            navigation_state = (
                "TURN_TO_WAYPOINT"
            )


        # -------------------------------------------------
        # CONTINUE FORWARD
        # -------------------------------------------------

        else:

            left_wheel_motor.setVelocity(
                FORWARD_WHEEL_SPEED
            )

            right_wheel_motor.setVelocity(
                FORWARD_WHEEL_SPEED
            )


    # =====================================================
    # GOAL REACHED
    # =====================================================

    elif (
        navigation_state
        == "GOAL_REACHED"
    ):

        left_wheel_motor.setVelocity(
            0.0
        )

        right_wheel_motor.setVelocity(
            0.0
        )


    # =====================================================
    # STOPPED
    # =====================================================

    elif (
        navigation_state
        == "STOPPED"
    ):

        left_wheel_motor.setVelocity(
            0.0
        )

        right_wheel_motor.setVelocity(
            0.0
        )


    # =====================================================
    # DEBUG OUTPUT
    # =====================================================

    if (
        len(
            navigation_waypoints
        )
        > 0
        and
        current_waypoint_index
        <
        len(
            navigation_waypoints
        )
    ):

        (
            current_target_x_position,
            current_target_y_position
        ) = navigation_waypoints[
            current_waypoint_index
        ]


        current_target_description = (
            f"Target=("
            f"{current_target_x_position:.2f},"
            f"{current_target_y_position:.2f})"
        )


        displayed_waypoint_number = (
            current_waypoint_index + 1
        )


    else:

        current_target_description = (
            "Target=GOAL"
        )

        displayed_waypoint_number = (
            len(
                navigation_waypoints
            )
        )


    print(
        f"X={robot_world_x_position:.3f}, "
        f"Y={robot_world_y_position:.3f}, "
        f"Yaw={robot_current_yaw:.3f}, "
        f"Left={left_obstacle_distance_value:.1f}, "
        f"Front={front_obstacle_distance_value:.1f}, "
        f"Right={right_obstacle_distance_value:.1f}, "
        f"Waypoint="
        f"{displayed_waypoint_number}/"
        f"{len(navigation_waypoints)}, "
        f"{current_target_description}, "
        f"TemporaryObstacles="
        f"{len(temporary_obstacles)}, "
        f"State={navigation_state}"
    )