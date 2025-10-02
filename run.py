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

def compare_centroids(calculated, answers, tolerance=1e-4):
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
    input_file = "inputs/random-n2048-d16-c16.txt"
    args = [
        "-i", input_file,                # Input file
        "-k", "16",                       # Number of clusters
        "-d", "16",                       # Dimensions of data
        "-e", "cuda",                    # Execution method: cuda, seq, or thrust
        "-t", "0.000000001",                  # Convergence threshold
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

            # --- Compare with Answer File ---
            answer_file = input_file.replace(".txt", "-answer.txt").replace("inputs","answers")
            answer_file = input_file.replace(".txt", "-answer.txt").replace("inputs/", "answers/")
            print(f"\n--- Validating against {answer_file} ---")
            answer_centroids = read_answer_file(answer_file)
            if answer_centroids:
                compare_centroids(final_centroids, answer_centroids)

    except subprocess.CalledProcessError as e:
        print("\n>>> Execution Failed! <<<")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    build_project()
    run_executable()
    print("\n--- Script Finished Successfully ---")