"""A.M.A.S.O.N. Webots Controller

Autonomous Mobile AI System for Optimized Navigation

Features:
- Differential-drive control
- GPS localization
- InertialUnit heading control
- Three distance sensors
- A* path planning
- Static obstacle mapping
- Waypoint following
- Finite-state-machine navigation
- Reactive obstacle avoidance
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

gps_sensor.enable(SIMULATION_TIME_STEP)

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

# With the sensor lookup table:
# 0 meters   = 0
# 1 meter    = 1000
#
# 300 is approximately 30 centimeters.
OBSTACLE_DETECTION_THRESHOLD = 300

# Robot must be within approximately 5 degrees of the
# desired heading before driving toward a waypoint.
WAYPOINT_HEADING_TOLERANCE = math.radians(5)

# Waypoint is reached when the robot is within 10 cm.
WAYPOINT_DISTANCE_TOLERANCE = 0.10

# Emergency/reactive avoidance turn.
OBSTACLE_AVOIDANCE_TURN_ANGLE = math.radians(30)


# =========================================================
# ARENA / GRID SETTINGS
# =========================================================

ARENA_SIZE_METERS = 5.0

GRID_CELL_SIZE_METERS = 0.25

GRID_CELL_COUNT = int(
    ARENA_SIZE_METERS / GRID_CELL_SIZE_METERS
)

ARENA_MINIMUM_COORDINATE = (
    -ARENA_SIZE_METERS / 2.0
)


# =========================================================
# STATIC OBSTACLE SETTINGS
# =========================================================
#
# Box:
# translation 1.2 0.4 0.15
#
# Box2:
# translation 0.7 1.1 0.15
#
# These size values assume both Webots Box geometries are:
#
# size 0.2 0.4 0.3
#
# Only X and Y matter for the A* map.
# =========================================================

BOX_ONE_CENTER_X = 1.20
BOX_ONE_CENTER_Y = 0.40

BOX_ONE_SIZE_X = 0.20
BOX_ONE_SIZE_Y = 0.40


BOX_TWO_CENTER_X = 0.70
BOX_TWO_CENTER_Y = 1.10

BOX_TWO_SIZE_X = 0.20
BOX_TWO_SIZE_Y = 0.40


# Extra clearance around each obstacle.
#
# This prevents A* from planning a path where the center
# of the robot technically clears the box but the chassis
# or wheels are too close.
STATIC_OBSTACLE_SAFETY_MARGIN = 0.30


# =========================================================
# START AND GOAL
# =========================================================

ROBOT_START_WORLD_POSITION = (
    0.0,
    0.0
)

ROBOT_GOAL_WORLD_POSITION = (
    1.7,
    1.20
)


# =========================================================
# ANGLE FUNCTIONS
# =========================================================

def normalize_angle_radians(angle_radians):
    """
    Normalize an angle to the range -pi through +pi.
    """

    while angle_radians > math.pi:
        angle_radians -= 2 * math.pi

    while angle_radians < -math.pi:
        angle_radians += 2 * math.pi

    return angle_radians


def calculate_angle_difference(
    current_angle_radians,
    starting_angle_radians
):
    """
    Calculate the normalized angular difference between
    a current orientation and a starting orientation.
    """

    angle_difference_radians = (
        current_angle_radians
        - starting_angle_radians
    )

    return normalize_angle_radians(
        angle_difference_radians
    )


# =========================================================
# GRID / WORLD CONVERSION
# =========================================================

def convert_world_position_to_grid_cell(
    world_x_position,
    world_y_position
):
    """
    Convert Webots world coordinates into grid coordinates.
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
    Convert a grid cell into the Webots world coordinates
    at the center of the cell.
    """

    world_x_position = (
        ARENA_MINIMUM_COORDINATE
        + (grid_column + 0.5)
        * GRID_CELL_SIZE_METERS
    )

    world_y_position = (
        ARENA_MINIMUM_COORDINATE
        + (grid_row + 0.5)
        * GRID_CELL_SIZE_METERS
    )

    return (
        world_x_position,
        world_y_position
    )


# =========================================================
# STATIC OBSTACLE MAPPING
# =========================================================

def mark_rectangular_obstacle_on_grid(
    occupancy_grid,
    obstacle_center_x,
    obstacle_center_y,
    obstacle_size_x,
    obstacle_size_y,
    safety_margin
):
    """
    Mark grid cells occupied by a rectangular obstacle.

    The safety margin expands the blocked area so that the
    robot's chassis and wheels maintain clearance.
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
    ) = convert_world_position_to_grid_cell(
        obstacle_minimum_x,
        obstacle_minimum_y
    )

    (
        maximum_grid_column,
        maximum_grid_row
    ) = convert_world_position_to_grid_cell(
        obstacle_maximum_x,
        obstacle_maximum_y
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


# =========================================================
# A* FUNCTIONS
# =========================================================

def calculate_manhattan_distance(
    first_grid_cell,
    second_grid_cell
):
    """
    Calculate Manhattan distance between two grid cells.
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

    Grid values:
        0 = open
        1 = obstacle
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

        if current_grid_cell == goal_grid_cell:

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
                or neighboring_grid_column
                >= GRID_CELL_COUNT
            ):
                continue

            if (
                neighboring_grid_row < 0
                or neighboring_grid_row
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
                < movement_cost_from_start[
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
                    + calculate_manhattan_distance(
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
# CREATE OCCUPANCY GRID
# =========================================================

occupancy_grid = [
    [
        0
        for grid_row
        in range(GRID_CELL_COUNT)
    ]
    for grid_column
    in range(GRID_CELL_COUNT)
]


# =========================================================
# ADD BOX ONE TO A* MAP
# =========================================================

mark_rectangular_obstacle_on_grid(
    occupancy_grid=occupancy_grid,
    obstacle_center_x=BOX_ONE_CENTER_X,
    obstacle_center_y=BOX_ONE_CENTER_Y,
    obstacle_size_x=BOX_ONE_SIZE_X,
    obstacle_size_y=BOX_ONE_SIZE_Y,
    safety_margin=STATIC_OBSTACLE_SAFETY_MARGIN
)


# =========================================================
# ADD BOX TWO TO A* MAP
# =========================================================

mark_rectangular_obstacle_on_grid(
    occupancy_grid=occupancy_grid,
    obstacle_center_x=BOX_TWO_CENTER_X,
    obstacle_center_y=BOX_TWO_CENTER_Y,
    obstacle_size_x=BOX_TWO_SIZE_X,
    obstacle_size_y=BOX_TWO_SIZE_Y,
    safety_margin=STATIC_OBSTACLE_SAFETY_MARGIN
)


# =========================================================
# START AND GOAL GRID CELLS
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


# =========================================================
# CALCULATE A* PATH
# =========================================================

planned_grid_path = calculate_astar_path(
    occupancy_grid,
    starting_grid_cell,
    goal_grid_cell
)


# =========================================================
# CREATE WORLD-SPACE WAYPOINTS
# =========================================================

navigation_waypoints = []

if planned_grid_path is not None:

    # Skip the first cell because the robot already
    # occupies the starting grid cell.
    for path_grid_cell in planned_grid_path[1:]:

        world_space_waypoint = (
            convert_grid_cell_to_world_position(
                path_grid_cell[0],
                path_grid_cell[1]
            )
        )

        navigation_waypoints.append(
            world_space_waypoint
        )

    # Use the exact requested goal for the final waypoint.
    if len(navigation_waypoints) > 0:

        navigation_waypoints[-1] = (
            ROBOT_GOAL_WORLD_POSITION
        )


# =========================================================
# PRINT PLANNED PATH
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

print(
    "Start cell:",
    starting_grid_cell
)

print(
    "Goal cell:",
    goal_grid_cell
)

print()

if planned_grid_path is None:

    print(
        "ERROR: A* COULD NOT FIND A PATH"
    )

else:

    print("A* PATH FOUND")

    print(
        "Grid cells:",
        len(planned_grid_path)
    )

    print(
        "Navigation waypoints:",
        len(navigation_waypoints)
    )

    print()

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
    or len(navigation_waypoints) == 0
):

    navigation_state = "STOPPED"

else:

    navigation_state = "TURN_TO_WAYPOINT"


current_waypoint_index = 0

obstacle_avoidance_start_yaw = 0.0


# =========================================================
# MAIN CONTROL LOOP
# =========================================================

while (
    amason_robot.step(
        SIMULATION_TIME_STEP
    )
    != -1
):

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
    # CHECK FOR COMPLETED NAVIGATION
    # =====================================================

    if (
        current_waypoint_index
        >= len(navigation_waypoints)
    ):

        navigation_state = (
            "GOAL_REACHED"
        )


    # =====================================================
    # REACTIVE OBSTACLE OVERRIDE
    # =====================================================
    #
    # IMPORTANT:
    #
    # Only trigger reactive avoidance while the robot is
    # actually driving.
    #
    # Previously the sensors could detect a box while the
    # robot was rotating toward a waypoint. That caused:
    #
    # TURN_TO_WAYPOINT
    # -> AVOID
    # -> TURN_TO_WAYPOINT
    # -> AVOID
    #
    # and the robot could become trapped in a loop.
    # =====================================================

    if navigation_state == "DRIVE_TO_WAYPOINT":

        # -------------------------------------------------
        # OBSTACLE DIRECTLY AHEAD
        # -------------------------------------------------

        if (
            front_obstacle_distance_value
            < OBSTACLE_DETECTION_THRESHOLD
        ):

            obstacle_avoidance_start_yaw = (
                robot_current_yaw
            )

            if (
                left_obstacle_distance_value
                > right_obstacle_distance_value
            ):

                navigation_state = (
                    "AVOID_LEFT"
                )

            else:

                navigation_state = (
                    "AVOID_RIGHT"
                )


        # -------------------------------------------------
        # OBSTACLE ON LEFT
        # -------------------------------------------------

        elif (
            left_obstacle_distance_value
            < OBSTACLE_DETECTION_THRESHOLD
        ):

            obstacle_avoidance_start_yaw = (
                robot_current_yaw
            )

            navigation_state = (
                "AVOID_RIGHT"
            )


        # -------------------------------------------------
        # OBSTACLE ON RIGHT
        # -------------------------------------------------

        elif (
            right_obstacle_distance_value
            < OBSTACLE_DETECTION_THRESHOLD
        ):

            obstacle_avoidance_start_yaw = (
                robot_current_yaw
            )

            navigation_state = (
                "AVOID_LEFT"
            )


    # =====================================================
    # TURN TOWARD WAYPOINT
    # =====================================================

    if navigation_state == "TURN_TO_WAYPOINT":

        (
            waypoint_target_x_position,
            waypoint_target_y_position
        ) = navigation_waypoints[
            current_waypoint_index
        ]

        distance_to_waypoint_x = (
            waypoint_target_x_position
            - robot_world_x_position
        )

        distance_to_waypoint_y = (
            waypoint_target_y_position
            - robot_world_y_position
        )

        straight_line_distance_to_waypoint = (
            math.sqrt(
                distance_to_waypoint_x
                * distance_to_waypoint_x
                +
                distance_to_waypoint_y
                * distance_to_waypoint_y
            )
        )


        # -------------------------------------------------
        # WAYPOINT ALREADY REACHED
        # -------------------------------------------------

        if (
            straight_line_distance_to_waypoint
            <= WAYPOINT_DISTANCE_TOLERANCE
        ):

            current_waypoint_index += 1

            if (
                current_waypoint_index
                >= len(navigation_waypoints)
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
                    - robot_current_yaw
                )
            )


            # ---------------------------------------------
            # HEADING CORRECT
            # ---------------------------------------------

            if (
                abs(waypoint_heading_error)
                <= WAYPOINT_HEADING_TOLERANCE
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
            # TARGET IS LEFT
            # ---------------------------------------------

            elif waypoint_heading_error > 0:

                left_wheel_motor.setVelocity(
                    -TURNING_WHEEL_SPEED
                )

                right_wheel_motor.setVelocity(
                    TURNING_WHEEL_SPEED
                )


            # ---------------------------------------------
            # TARGET IS RIGHT
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

    elif navigation_state == "DRIVE_TO_WAYPOINT":

        (
            waypoint_target_x_position,
            waypoint_target_y_position
        ) = navigation_waypoints[
            current_waypoint_index
        ]

        distance_to_waypoint_x = (
            waypoint_target_x_position
            - robot_world_x_position
        )

        distance_to_waypoint_y = (
            waypoint_target_y_position
            - robot_world_y_position
        )

        straight_line_distance_to_waypoint = (
            math.sqrt(
                distance_to_waypoint_x
                * distance_to_waypoint_x
                +
                distance_to_waypoint_y
                * distance_to_waypoint_y
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
                - robot_current_yaw
            )
        )


        # -------------------------------------------------
        # WAYPOINT REACHED
        # -------------------------------------------------

        if (
            straight_line_distance_to_waypoint
            <= WAYPOINT_DISTANCE_TOLERANCE
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
                >= len(navigation_waypoints)
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
            abs(waypoint_heading_error)
            > WAYPOINT_HEADING_TOLERANCE
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
        # CONTINUE DRIVING
        # -------------------------------------------------

        else:

            left_wheel_motor.setVelocity(
                FORWARD_WHEEL_SPEED
            )

            right_wheel_motor.setVelocity(
                FORWARD_WHEEL_SPEED
            )


    # =====================================================
    # AVOID OBSTACLE BY TURNING LEFT
    # =====================================================

    elif navigation_state == "AVOID_LEFT":

        left_wheel_motor.setVelocity(
            -TURNING_WHEEL_SPEED
        )

        right_wheel_motor.setVelocity(
            TURNING_WHEEL_SPEED
        )

        obstacle_avoidance_angle_turned = abs(
            calculate_angle_difference(
                robot_current_yaw,
                obstacle_avoidance_start_yaw
            )
        )

        if (
            obstacle_avoidance_angle_turned
            >= OBSTACLE_AVOIDANCE_TURN_ANGLE
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


    # =====================================================
    # AVOID OBSTACLE BY TURNING RIGHT
    # =====================================================

    elif navigation_state == "AVOID_RIGHT":

        left_wheel_motor.setVelocity(
            TURNING_WHEEL_SPEED
        )

        right_wheel_motor.setVelocity(
            -TURNING_WHEEL_SPEED
        )

        obstacle_avoidance_angle_turned = abs(
            calculate_angle_difference(
                robot_current_yaw,
                obstacle_avoidance_start_yaw
            )
        )

        if (
            obstacle_avoidance_angle_turned
            >= OBSTACLE_AVOIDANCE_TURN_ANGLE
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


    # =====================================================
    # GOAL REACHED
    # =====================================================

    elif navigation_state == "GOAL_REACHED":

        left_wheel_motor.setVelocity(
            0.0
        )

        right_wheel_motor.setVelocity(
            0.0
        )


    # =====================================================
    # NO PATH
    # =====================================================

    elif navigation_state == "STOPPED":

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
        current_waypoint_index
        < len(navigation_waypoints)
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

        displayed_waypoint_number = len(
            navigation_waypoints
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
        f"State={navigation_state}"
    )