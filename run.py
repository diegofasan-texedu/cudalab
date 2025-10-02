#!/usr/bin/env python3

import subprocess
import sys
import os
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

def build_project():
    """Runs the 'make' command to build the project."""
    print("--- Building Project ---")
    try:
        # We use capture_output=True to hide the make command's output unless there's an error.
        result = subprocess.run(["make"], check=True, capture_output=True, text=True)
        print("Build successful.")
        # print(result.stdout) # Uncomment to see build output
    except FileNotFoundError:
        print("Error: 'make' command not found. Is make installed and in your PATH?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(">>> Build Failed! <<<")
        print("--- stdout ---")
        print(e.stdout)
        print("--- stderr ---")
        print(e.stderr)
        sys.exit(1)

def parse_centroids(output_text):
    """Parses the stdout of the k-means executable to extract centroid data."""
    centroids = []
    # Regex to find lines with centroid data and capture the numbers inside the brackets.
    # It looks for "Centroid <number>: [<numbers>]"
    centroid_pattern = re.compile(r"^\s*Centroid \d+:\s*\[(.*)\]")

    for line in output_text.splitlines():
        match = centroid_pattern.match(line)
        if match:
            # The first group captures the comma-separated numbers.
            numbers_str = match.group(1)
            # Split the string by comma and convert each part to a float.
            try:
                point = [float(n) for n in numbers_str.split(',')]
                centroids.append(point)
            except ValueError:
                print(f"Warning: Could not parse numbers in line: {line}")
    return centroids

def read_answer_file(filepath):
    """Reads a file of points and returns a list of lists."""
    if not os.path.exists(filepath):
        print(f"Error: Answer file not found at '{filepath}'")
        return None
    
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # The answer file might have a leading index like the input file.
                # We split the line and take all but the first element.
                point = [float(n) for n in line.split()[1:]]
                points.append(point)
            except ValueError:
                print(f"Warning: Could not parse numbers in answer line: {line}")
    return points

def compare_centroids(calculated, answers, tolerance=1e-6):
    """
    Compares two sets of centroids for similarity.

    This function handles the case where the order of centroids is different
    by finding the optimal one-to-one mapping between the two sets that
    minimizes the sum of Euclidean distances.
    """
    if len(calculated) != len(answers):
        print(f"Validation Failed: Mismatched number of centroids. Got {len(calculated)}, expected {len(answers)}.")
        return False

    # Create a cost matrix where cost[i, j] is the distance between
    # calculated[i] and answers[j].
    calculated_np = np.array(calculated)
    print(calculated_np)
    answers_np = np.array(answers)
    print(answers_np)
    cost_matrix = np.linalg.norm(calculated_np[:, np.newaxis, :] - answers_np[np.newaxis, :, :], axis=2)

    # Use the Hungarian algorithm to find the optimal assignment (best pairing).
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Check if the distance for each matched pair is within the tolerance.
    total_distance = 0
    for r, c in zip(row_ind, col_ind):
        distance = cost_matrix[r, c]
        total_distance += distance
        if distance > tolerance:
            print(f"Validation Failed: Centroid pair distance ({distance:.6f}) exceeds tolerance ({tolerance}).")
            return False
    
    print(f"Validation Successful! All centroid pairs are within tolerance. Total distance: {total_distance:.6f}")
    return True

def validate_point_assignments(points, final_centroids, threshold=1e-5):
    """
    Validates the k-means result based on the user's specified method.
    """
    print("\n--- Validating Point Assignments ---")
    points_np = np.array(points)
    centroids_np = np.array(final_centroids)

    # 1. For each point, find its closest centroid from the final set.
    # This gives us the 'optimal' assignment for each point.
    dist_sq = np.sum((points_np[:, np.newaxis, :] - centroids_np[np.newaxis, :, :])**2, axis=2)
    dis0 = np.min(dist_sq, axis=1) # dis0 is the distance to the closest centroid.

    # 2. Re-calculate centroids based on these optimal assignments.
    optimal_assignments = np.argmin(dist_sq, axis=1)
    num_clusters = len(centroids_np)
    recalculated_centroids = np.array([points_np[optimal_assignments == k].mean(axis=0) for k in range(num_clusters)])

    # 3. For each point, find the distance to its re-calculated centroid.
    dis1 = np.array([np.sum((points_np[i] - recalculated_centroids[optimal_assignments[i]])**2) for i in range(len(points_np))])

    # 4. Check if dis0 and dis1 are close.
    diff = np.abs(dis0 - dis1)
    wrong_points = np.where(diff > threshold)[0]

    num_wrong = len(wrong_points)
    if num_wrong == 0:
        print(f"Assignment Validation Successful! All {len(points_np)} points are optimally placed.")
    else:
        print(f"Assignment Validation Failed: Found {num_wrong} non-optimally placed points.")
        for i, point_idx in enumerate(wrong_points[:5]):
            print(f"  - Point {point_idx}: dis0_sq={dis0[point_idx]:.6f}, dis1_sq={dis1[point_idx]:.6f}, diff={diff[point_idx]:.6f}")
        if num_wrong > 5:
            print("  ...")
    return num_wrong

def compare_centroids_legacy(calculated, answers, tolerance=1e-6):
    """
    Compares two sets of centroids for similarity.

    This function handles the case where the order of centroids is different
    by finding the optimal one-to-one mapping between the two sets that
    minimizes the sum of Euclidean distances.
    """
    if len(calculated) != len(answers):
        print(f"Validation Failed: Mismatched number of centroids. Got {len(calculated)}, expected {len(answers)}.")
        return False

    # Create a cost matrix where cost[i, j] is the distance between
    # calculated[i] and answers[j].
    calculated_np = np.array(calculated)
    print(calculated_np)
    answers_np = np.array(answers)
    print(answers_np)
    cost_matrix = np.linalg.norm(calculated_np[:, np.newaxis, :] - answers_np[np.newaxis, :, :], axis=2)

    # Use the Hungarian algorithm to find the optimal assignment (best pairing).
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Check if the distance for each matched pair is within the tolerance.
    total_distance = 0
    for r, c in zip(row_ind, col_ind):
        distance = cost_matrix[r, c]
        total_distance += distance
        if distance > tolerance:
            print(f"Validation Failed: Centroid pair distance ({distance:.6f}) exceeds tolerance ({tolerance}).")
            return False
    
    print(f"Validation Successful! All centroid pairs are within tolerance. Total distance: {total_distance:.6f}")
    return True

def run_executable():
    """Runs the compiled k-means executable with example arguments."""
    print("\n--- Running K-Means Executable ---")
    
    executable_path = "./bin/kmeans"
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at '{executable_path}'.")
        sys.exit(1)

    # --- CONFIGURE YOUR K-MEANS ARGUMENTS HERE ---
    # This example assumes an input file at 'data/points_2d_1000.txt'
    # with 2 dimensions. Adjust these values for your dataset.
    input_file = "inputs/random-n65536-d32-c16.txt"
    args = [
        "-i", input_file,                # Input file
        "-k", "16",                       # Number of clusters
        "-d", "32",                       # Dimensions of data
        "-e", "cuda",                    # Execution method: cuda, seq, or thrust
        "-t", "0.000001",                  # Convergence threshold
        "-m", "150",                     # Max iterations
        "-s", "20",                            # Output final centroids
        # "-n", "500",                     # Max iterations
        # "-o",                            # Output final centroids
        "-v"                             # Verbose mode
    ]

    command = [executable_path] + args
    print(f"Executing: {' '.join(command)}\n")

    try:
        # Run the command and stream its output directly to the console.
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Print the full output from the C++ program
        print(result.stdout)
        
        # Parse the output to extract the centroids
        final_centroids = parse_centroids(result.stdout)
        
        if final_centroids:
            print("\n--- Centroids Captured in Python ---")
            for i, centroid in enumerate(final_centroids):
                # Truncate for display purposes
                centroid_str = ", ".join([f"{x:.4f}" for x in centroid[:4]])
                print(f"  Centroid {i}: [{centroid_str}, ...]")

            # --- Load original points for validation ---
            points = read_answer_file(input_file) # Using read_answer_file as it reads points

            # --- New Validation Method as Requested ---
            if points:
                validate_point_assignments(points, final_centroids)

            # --- Original Centroid Comparison (Optional) ---
            answer_file = input_file.replace(".txt", "-answer.txt").replace("inputs","answers")
            answer_file = input_file.replace(".txt", "-answer.txt").replace("inputs/", "answers/")
            print(f"\n--- Comparing Final Centroids against Answer File: {answer_file} ---")
            answer_centroids = read_answer_file(answer_file)
            if answer_centroids:
                compare_centroids_legacy(final_centroids, answer_centroids)

    except subprocess.CalledProcessError as e:
        print("\n>>> Execution Failed! <<<")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    build_project()
    run_executable()
    print("\n--- Script Finished Successfully ---")