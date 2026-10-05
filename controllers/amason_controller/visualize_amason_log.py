"""A.M.A.S.O.N. Navigation Log Visualizer

Run after a Webots simulation:

    python visualize_amason_log.py

Or provide a CSV path:

    python visualize_amason_log.py path/to/amason_navigation_log.csv
"""

import csv
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_PATH = os.path.join(SCRIPT_DIRECTORY, "amason_navigation_log.csv")

STATIC_OBSTACLES = [
    (1.20, 0.40, 0.20, 0.40, "K1"),
    (0.70, 1.10, 0.20, 0.40, "K2"),
    (-1.20, -1.20, 0.22, 0.22, "K3"),
    (0.00, -1.25, 0.22, 0.22, "K4"),
    (1.20, -1.10, 0.22, 0.22, "K5"),
    (-1.65, -0.10, 0.22, 0.22, "K6"),
    (-0.45, -0.05, 0.22, 0.22, "K7"),
    (-1.15, 1.10, 0.22, 0.22, "K8"),
    (0.10, 1.70, 0.22, 0.22, "K9"),
    (1.55, 1.50, 0.22, 0.22, "K10"),
    (-0.45, 0.85, 0.22, 0.22, "K11"),
    (-1.65, 1.65, 0.22, 0.22, "K12"),
]

PATROL_ENABLED = False
SINGLE_GOAL_WORLD_POSITION = (2.0, 2.0)

# Retained so the same visualizer can be switched back to patrol mode later.
PATROL_LOCATIONS = [
    (1.80, 1.20),
    (-1.50, 1.50),
    (-1.50, -1.50),
    (1.50, -1.50),
]


def read_navigation_log(log_path):
    with open(log_path, "r", newline="") as log_file:
        return list(csv.DictReader(log_file))


def numeric_column(rows, column_name):
    return [float(row[column_name]) for row in rows]


def print_performance_summary(rows):
    final_row = rows[-1]
    print()
    print("==========================================")
    print("A.M.A.S.O.N. PERFORMANCE SUMMARY")
    print("==========================================")
    print("Simulation time:", f'{float(final_row["Time"]):.2f} seconds')
    print("Distance traveled:", f'{float(final_row["DistanceTraveledMeters"]):.2f} meters')
    print("Path plans:", final_row["PathPlanCount"])
    print("Dynamic replans:", final_row["DynamicReplanCount"])
    print("Heading corrections:", final_row["HeadingCorrectionCount"])
    if PATROL_ENABLED:
        print("Patrol points reached:", final_row["PatrolPointsReached"])
    else:
        print("Navigation mode: corner-to-corner single goal")
    print("==========================================")


def create_route_plot(rows):
    x_positions = numeric_column(rows, "X")
    y_positions = numeric_column(rows, "Y")

    figure, axes = plt.subplots()
    axes.plot(x_positions, y_positions, label="Robot trajectory")
    axes.scatter([x_positions[0]], [y_positions[0]], marker="o", label="Start")

    for center_x, center_y, size_x, size_y, name in STATIC_OBSTACLES:
        obstacle_rectangle = Rectangle(
            (center_x - size_x / 2.0, center_y - size_y / 2.0),
            size_x,
            size_y,
            fill=False,
        )
        axes.add_patch(obstacle_rectangle)
        axes.text(center_x, center_y, name)

    if PATROL_ENABLED:
        for patrol_number, patrol_location in enumerate(PATROL_LOCATIONS, start=1):
            axes.scatter([patrol_location[0]], [patrol_location[1]], marker="x")
            axes.text(patrol_location[0], patrol_location[1], f"P{patrol_number}")
    else:
        axes.scatter(
            [SINGLE_GOAL_WORLD_POSITION[0]],
            [SINGLE_GOAL_WORLD_POSITION[1]],
            marker="x",
            label="Goal",
        )
        axes.text(
            SINGLE_GOAL_WORLD_POSITION[0],
            SINGLE_GOAL_WORLD_POSITION[1],
            "Goal",
        )

    temporary_obstacle_x = []
    temporary_obstacle_y = []

    for row in rows:
        if (
            "TEMPORARY_OBSTACLE_ADDED" in row["Event"]
            and row["EventX"]
            and row["EventY"]
        ):
            temporary_obstacle_x.append(float(row["EventX"]))
            temporary_obstacle_y.append(float(row["EventY"]))

    if temporary_obstacle_x:
        axes.scatter(
            temporary_obstacle_x,
            temporary_obstacle_y,
            marker="s",
            label="Detected temporary obstacles",
        )

    axes.set_title("A.M.A.S.O.N. Corner-to-Corner Navigation Trajectory")
    axes.set_xlabel("World X (meters)")
    axes.set_ylabel("World Y (meters)")
    axes.set_xlim(-2.5, 2.5)
    axes.set_ylim(-2.5, 2.5)
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)
    axes.legend()
    figure.tight_layout()


def create_sensor_plot(rows):
    times = numeric_column(rows, "Time")
    left_values = numeric_column(rows, "LeftSensor")
    front_values = numeric_column(rows, "FrontSensor")
    right_values = numeric_column(rows, "RightSensor")

    figure, axes = plt.subplots()
    axes.plot(times, left_values, label="Left sensor")
    axes.plot(times, front_values, label="Front sensor")
    axes.plot(times, right_values, label="Right sensor")
    axes.set_title("A.M.A.S.O.N. Distance Sensor History")
    axes.set_xlabel("Simulation Time (seconds)")
    axes.set_ylabel("Sensor Value")
    axes.grid(True)
    axes.legend()
    figure.tight_layout()


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_PATH

    if not os.path.exists(log_path):
        print("Navigation log not found:", log_path)
        return

    rows = read_navigation_log(log_path)

    if not rows:
        print("Navigation log contains no data.")
        return

    print_performance_summary(rows)
    create_route_plot(rows)
    create_sensor_plot(rows)
    plt.show()


if __name__ == "__main__":
    main()
