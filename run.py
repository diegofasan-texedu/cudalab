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

    # --- Print the full cost matrix for inspection ---
    print("\n--- Cost Matrix (Euclidean Distances) ---")
    print("Each row is a calculated centroid, each column is an answer centroid.")
    print(np.array2string(cost_matrix, formatter={'float_kind':lambda x: "%.6f" % x}))
    
 
    # Use the Hungarian algorithm to find the optimal assignment (best pairing).
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
 
    # Print both sets of centroids for inspection before showing the distances.
    print_centroid_list("Final Centroids (Calculated)", calculated)
    print_centroid_list("Answer Centroids (Ground Truth)", answers)

    # --- Display the distance for each matched pair ---
    print("\n--- Distances between Matched Final and Answer Centroids ---")
    distances = cost_matrix[row_ind, col_ind]
    for i in range(len(row_ind)): # type: ignore
        # calculated_idx -> answer_idx
        print(f"  Calculated Centroid {row_ind[i]:<2} <-> Answer Centroid {col_ind[i]:<2} | Distance: {distances[i]:.6f}")
 
    # Check if the distance for each matched pair is within the tolerance.
    max_distance = distances.max()
 
    if max_distance > tolerance:
        print(f"\nValidation Failed: Maximum centroid distance ({max_distance:.6f}) exceeds tolerance ({tolerance}).")
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
        "-s", "8675309",                            # Output final centroids
        "-v",                             # Verbose mode
        "-c"
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

            # --- Load original points and answer centroids for validation ---
            answer_file = input_file.replace(".txt", "-answer.txt").replace("inputs/", "answers/")
            answer_centroids = read_points_file(answer_file, has_header=False)
            
            if answer_centroids:
                # 1. Compare the final centroids directly to the answer centroids
                print(f"\n--- Comparing Final Centroids against Answer File: {answer_file} ---")
                compare_centroids(final_centroids, answer_centroids, tolerance=1e-5)
                # 2. Validate the stability of the point assignments
                points = read_points_file(input_file, has_header=True)
                validate_point_assignments(points, final_centroids, answer_centroids)

    except subprocess.CalledProcessError as e:
        print("\n>>> Execution Failed! <<<")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    build_project()
    run_executable()
    print("\n--- Script Finished Successfully ---")