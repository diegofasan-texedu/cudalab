#!/usr/bin/env python3

import subprocess
import sys
import os
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

def build_project():
    """Runs 'make all' to build both float and double versions of the project."""
    print("--- Building Project (both float and double) ---")
    try:
        # We use capture_output=True to hide the make command's output unless there's an error.
        result = subprocess.run(["make", "all"], check=True, capture_output=True, text=True)
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
    """
    Parses the stdout of the k-means executable to extract centroid data.
    The expected format is:
    ...
    <iterations>,<time_per_iter>
    <cluster_id> <coord1> <coord2> ...
    <cluster_id> <coord1> <coord2> ...
    ...
    """
    centroids = []
    # Regex to find a line that looks like a centroid: an integer followed by floats.
    # This is flexible and avoids matching other text lines.
    centroid_pattern = re.compile(r"^\s*\d+\s+[\d\.\-e\+]+")

    for line in output_text.splitlines():
        # Check if the line matches the pattern of a centroid data line.
        if centroid_pattern.match(line.strip()):
            try:
                # Split the line by whitespace and convert all but the first element (cluster ID) to float.
                point = [float(n) for n in line.strip().split()[1:]]
                centroids.append(point)
            except ValueError:
                print(f"Warning: Could not parse numbers in line: {line}")
    return centroids

def read_points_file(filepath, has_header=False):
    """
    Reads a file of points (or centroids) and returns a list of lists.
    
    Args:
        filepath (str): The path to the file.
        has_header (bool): If True, skips the first line (assumed to be a point count).
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None
    
    points = []
    with open(filepath, 'r') as f:
        try:
            if has_header:
                f.readline() # Skip header line

            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    # Files have a leading index for each point/centroid.
                    # We split the line and take all but the first element.
                    point = [float(n) for n in line.split()[1:]]
                    points.append(point)
                except (ValueError, IndexError):
                    line_num = i + (2 if has_header else 1)
                    print(f"Warning: Could not parse numbers in file '{filepath}' on line {line_num}: {line}")
        except (IOError, ValueError) as e:
            print(f"Error reading point file '{filepath}': {e}")
    return points

def print_centroid_list(name, centroids):
    """Helper function to print a list of centroids with formatting."""
    print(f"\n--- {name} (CSV Format, double precision) ---")
    for i, centroid in enumerate(centroids):
        # Create a list of strings for all coordinates, formatted to 12 decimal places for doubles.
        coords_str_list = [f"{x:.12f}" for x in centroid]
        # Join the index and all coordinates with a comma to create a CSV line.
        csv_line = ",".join([str(i)] + coords_str_list)
        print(csv_line)

def compare_centroids(calculated, answers, tolerance=1e-4):
    """
    Compares two sets of centroids for similarity.
 
    This function handles the case where the order of centroids is different
    by finding the optimal one-to-one mapping between the two sets that
    minimizes the sum of Euclidean distances. Each matched pair must have a
    distance below the specified tolerance for the validation to pass.
    """
    if len(calculated) != len(answers):
        print(f"Validation Failed: Mismatched number of centroids. Got {len(calculated)}, expected {len(answers)}.")
        return False
 
    # Create a cost matrix where cost[i, j] is the distance between
    # calculated[i] and answers[j].
    calculated_np = np.array(calculated)
    answers_np = np.array(answers)
    cost_matrix = np.linalg.norm(calculated_np[:, np.newaxis, :] - answers_np[np.newaxis, :, :], axis=2)

    # Use the Hungarian algorithm to find the optimal assignment (best pairing).
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
 
    distances = cost_matrix[row_ind, col_ind]
 
    # Check if the distance for each matched pair is within the tolerance.
    max_distance = distances.max()
 
    if max_distance > tolerance:
        print(f"\nValidation Failed: Maximum centroid distance ({max_distance:.6f}) exceeds tolerance ({tolerance}).")
        print("--- Centroid Pairs Exceeding Tolerance ---")
        # Find and print details for all pairs that failed the check.
        failed_indices = np.where(distances > tolerance)[0]
        for i in failed_indices:
            calc_idx = row_ind[i]
            ans_idx = col_ind[i]
            dist = distances[i]
            print(f"  - Calculated Centroid {calc_idx} <-> Answer Centroid {ans_idx} | Distance: {dist:.6f}")
            # print(f"    Calculated: {calculated[calc_idx]}") # Uncomment for full coordinate details
            # print(f"    Answer:     {answers[ans_idx]}")     # Uncomment for full coordinate details
        return False
 
    print(f"\nValidation Successful! All matched centroid pairs are within tolerance. Max distance: {max_distance:.6f}")
    return True

def validate_point_assignments(points, final_centroids, answer_centroids, tolerance=1e-5):
    """
    Validates the clustering quality by comparing the distance from each point
    to its closest calculated centroid versus its closest answer centroid.
    A point is "wrong" if this difference exceeds a tolerance.
    """
    print("\n--- Validating Clustering Quality against Answer ---")
    points_np = np.array(points)
    final_centroids_np = np.array(final_centroids)
    answer_centroids_np = np.array(answer_centroids)
    num_points = len(points_np)

    # 1. For each point, find the distance to its closest CALCULATED centroid.
    dist_to_final = np.linalg.norm(points_np[:, np.newaxis, :] - final_centroids_np[np.newaxis, :, :], axis=2)
    min_dist_to_final = np.min(dist_to_final, axis=1)

    # 2. For each point, find the distance to its closest ANSWER centroid.
    dist_to_answer = np.linalg.norm(points_np[:, np.newaxis, :] - answer_centroids_np[np.newaxis, :, :], axis=2)
    min_dist_to_answer = np.min(dist_to_answer, axis=1)

    # 3. A point is "wrong" if the difference between these two minimum distances
    #    is larger than the tolerance.
    diff = np.abs(min_dist_to_final - min_dist_to_answer)
    wrong_points_indices = np.where(diff > tolerance)[0]

    num_wrong = len(wrong_points_indices)
    if num_wrong == 0:
        print(f"Quality Validation Successful! All {num_points} points have optimal distances within tolerance.")
    else:
        print(f"Quality Validation Failed: Found {num_wrong} points with suboptimal distances.")
        # Print details for up to 10 wrong points.
        for i, point_idx in enumerate(wrong_points_indices[:40]):
            print(f"  - Point {point_idx}: dist_to_final={min_dist_to_final[point_idx]:.6f}, dist_to_answer={min_dist_to_answer[point_idx]:.6f}, diff={diff[point_idx]:.6f}")
        if num_wrong > 10:
            print("  ...")
    
def run_and_average_performance(command, num_runs=10):
    """Runs the executable multiple times and averages the performance."""
    avg_iter_times = []
    perf_pattern = re.compile(r"^\s*(\d+),([\d\.\-e\+]+)")

    print(f"\n--- Running {num_runs} times to average performance ---")
    for i in range(num_runs):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            # Find the performance line (e.g., "iterations,avg_time_ms")
            for line in result.stdout.splitlines():
                match = perf_pattern.match(line)
                if match:
                    avg_time = float(match.group(2))
                    avg_iter_times.append(avg_time)
                    print(f"  Run {i + 1}/{num_runs}: {avg_time:.4f} ms/iter")
                    break # Found the line, move to next run
        except subprocess.CalledProcessError as e:
            print(f"Execution failed on run {i+1}: {e.stderr}")
            return

    if avg_iter_times:
        overall_avg = np.mean(avg_iter_times)
        print(f"\n--- Average Performance ---")
        print(f"Overall average time per iteration: {overall_avg:.4f} ms")

def run_executable(precision="double"):
    """Runs the compiled k-means executable with example arguments."""
    print("\n--- Running K-Means Executable ---")
    
    executable_path = f"./bin/kmeans_{precision}" if precision == "float" else "./bin/kmeans"
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at '{executable_path}'. Did you run 'make'?")
        sys.exit(1)

    # --- CONFIGURE YOUR K-MEANS ARGUMENTS HERE ---
    # This example assumes an input file at 'data/points_2d_1000.txt'
    # with 2 dimensions. Adjust these values for your dataset.
    input_file = "inputs/random-n2048-d16-c16.txt"
    dims = "16"
    # input_file = "inputs/random-n16384-d24-c16.txt"
    # dims = "24"
    # input_file = "inputs/random-n65536-d32-c16.txt"
    # dims = "32"
    args = [
        "-i", input_file,                # Input file
        "-k", "16",                      # Number of clusters
        "-d", dims,                      # Dimensions of data
        "-e", "thrust",                    # Execution method: cuda, seq, or thrust
        "-t", "0.0001",                  # Convergence threshold
        "-m", "200",                     # Max iterations
        "-s", "8675309",                 # Seed
        "-c" # Add -c to validate centroids. Remove for performance averaging.
    ]

    # --- Extract tolerance for Python validation functions ---
    # Find the value associated with the '-t' flag in the args list.
    tolerance = 1e-5 # Default tolerance if -t is not in args
    try:
        t_index = args.index("-t")
        if t_index + 1 < len(args):
            tolerance = float(args[t_index + 1])
    except (ValueError, IndexError):
        print("Warning: '-t' flag not found in args list. Using default tolerance for Python validation.")

    command = [executable_path] + args
    print(f"Executing: {' '.join(command)}\n")

    # --- Decide whether to run for performance or for validation ---
    if "-c" not in args:
        # If not validating centroids, run multiple times for performance measurement.
        run_and_average_performance(command, num_runs=20)
        return

    try:
        # Run the command and stream its output directly to the console.
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Print the full output from the C++ program
        print(result.stdout)
        
        # Parse the output to extract the centroids
        final_centroids = parse_centroids(result.stdout)
        
        if final_centroids:
            # --- Load original points and answer centroids for validation ---
            answer_file = input_file.replace(".txt", "-answer.txt").replace("inputs/", "answers/")
            answer_centroids = read_points_file(answer_file, has_header=False)
            
            if answer_centroids:
                # 1. Compare the final centroids directly to the answer centroids
                print(f"\n--- Comparing Final Centroids against Answer File: {answer_file} ---")
                compare_centroids(final_centroids, answer_centroids, tolerance=tolerance)
                # 2. Validate the stability of the point assignments
                points = read_points_file(input_file, has_header=True)
                validate_point_assignments(points, final_centroids, answer_centroids, tolerance=tolerance)

    except subprocess.CalledProcessError as e:
        print("\n>>> Execution Failed! <<<")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # --- Determine Precision from Command-Line Arguments ---
    # Usage: python3 run.py [float|double]
    # Defaults to "double" if no argument is provided.
    target_precision = "double"
    if len(sys.argv) > 1 and sys.argv[1].lower() == "float":
        target_precision = "float"

    # Build both versions
    build_project()
    
    # Run the selected version
    run_executable(precision=target_precision)
    
    print("\n--- Script Finished Successfully ---")