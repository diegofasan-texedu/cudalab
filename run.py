#!/usr/bin/env python3

import subprocess
import sys
import os
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

def build_project():
    """Builds both float and double versions of the project using 'make all'."""
    print("--- Building Project (both float and double) ---")
    try:
        result = subprocess.run(["make", "all"], check=True, capture_output=True, text=True)
        print("Build successful.")
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
    """Parses centroid data from the k-means executable's stdout."""
    centroids = []
    # Regex to find a line with a centroid: an integer followed by floats.
    centroid_pattern = re.compile(r"^\s*\d+\s+[\d\.\-e\+]+")

    for line in output_text.splitlines():
        if centroid_pattern.match(line.strip()):
            try:
                point = [float(n) for n in line.strip().split()[1:]]
                centroids.append(point)
            except ValueError:
                print(f"Warning: Could not parse numbers in line: {line}")
    return centroids

def read_points_file(filepath, has_header=False):
    """Reads a file of points/centroids, skipping the header if specified."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None
    
    points = []
    with open(filepath, 'r') as f:
        try:
            if has_header:
                f.readline()

            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    point = [float(n) for n in line.split()[1:]]
                    points.append(point)
                except (ValueError, IndexError):
                    line_num = i + (2 if has_header else 1)
                    print(f"Warning: Could not parse numbers in file '{filepath}' on line {line_num}: {line}")
        except (IOError, ValueError) as e:
            print(f"Error reading point file '{filepath}': {e}")
    return points

def print_centroid_list(name, centroids):
    """Prints a list of centroids in a formatted CSV style."""
    print(f"\n--- {name} (CSV Format, double precision) ---")
    for i, centroid in enumerate(centroids):
        coords_str_list = [f"{x:.12f}" for x in centroid]
        csv_line = ",".join([str(i)] + coords_str_list)
        print(csv_line)

def compare_centroids(calculated, answers, tolerance=1e-4):
    """Compares two sets of centroids, handling different ordering."""
    if len(calculated) != len(answers):
        print(f"Validation Failed: Mismatched number of centroids. Got {len(calculated)}, expected {len(answers)}.")
        return False
 
    # Create a cost matrix where cost[i, j] is the distance between calculated[i] and answers[j].
    calculated_np = np.array(calculated)
    answers_np = np.array(answers)
    cost_matrix = np.linalg.norm(calculated_np[:, np.newaxis, :] - answers_np[np.newaxis, :, :], axis=2)

    # Use the Hungarian algorithm to find the optimal assignment (best pairing).
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
 
    distances = cost_matrix[row_ind, col_ind]
    max_distance = distances.max()
 
    if max_distance > tolerance:
        print(f"\nValidation Failed: Maximum centroid distance ({max_distance:.6f}) exceeds tolerance ({tolerance}).")
        print("--- Centroid Pairs Exceeding Tolerance ---")
        failed_indices = np.where(distances > tolerance)[0]
        for i in failed_indices:
            calc_idx = row_ind[i]
            ans_idx = col_ind[i]
            dist = distances[i]
            print(f"  - Calculated Centroid {calc_idx} <-> Answer Centroid {ans_idx} | Distance: {dist:.6f}")
        return False
 
    print(f"\nValidation Successful! All matched centroid pairs are within tolerance. Max distance: {max_distance:.6f}")
    return True

def validate_point_assignments(points, final_centroids, answer_centroids, tolerance=1e-5):
    """
    Validates clustering quality by comparing point-to-centroid distances.
    """
    print("\n--- Validating Clustering Quality against Answer ---")
    points_np = np.array(points)
    final_centroids_np = np.array(final_centroids)
    answer_centroids_np = np.array(answer_centroids)
    num_points = len(points_np)

    # For each point, find the distance to its closest calculated centroid.
    dist_to_final = np.linalg.norm(points_np[:, np.newaxis, :] - final_centroids_np[np.newaxis, :, :], axis=2)
    min_dist_to_final = np.min(dist_to_final, axis=1)

    # For each point, find the distance to its closest answer centroid.
    dist_to_answer = np.linalg.norm(points_np[:, np.newaxis, :] - answer_centroids_np[np.newaxis, :, :], axis=2)
    min_dist_to_answer = np.min(dist_to_answer, axis=1)

    diff = np.abs(min_dist_to_final - min_dist_to_answer)
    wrong_points_indices = np.where(diff > tolerance)[0]

    num_wrong = len(wrong_points_indices)
    if num_wrong == 0:
        print(f"Quality Validation Successful! All {num_points} points have optimal distances within tolerance.")
    else:
        print(f"Quality Validation Failed: Found {num_wrong} points with suboptimal distances.")
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
            for line in result.stdout.splitlines():
                match = perf_pattern.match(line)
                if match:
                    avg_time = float(match.group(2))
                    avg_iter_times.append(avg_time)
                    print(f"  Run {i + 1}/{num_runs}: {avg_time:.4f} ms/iter")
                    break
        except subprocess.CalledProcessError as e:
            print(f"Execution failed on run {i+1}: {e.stderr}")
            return

    if avg_iter_times:
        overall_avg = np.mean(avg_iter_times)
        print(f"\n--- Average Performance ---")
        print(f"Overall average time per iteration: {overall_avg:.4f} ms")

def profile_executable(precision="double"):
    """Profiles the k-means executable using NVIDIA Nsight Systems."""
    print(f"\n--- Profiling K-Means Executable ({precision}) with Nsight Systems ---")

    executable_path = f"./bin/kmeans_{precision}" if precision == "float" else "./bin/kmeans"
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at '{executable_path}'. Did you run 'make'?")
        sys.exit(1)

    # --- K-MEANS ARGUMENTS (modify these to profile different scenarios) ---
    input_file = "inputs/random-n65536-d32-c16.txt"
    dims = "32"
    # Choose the method to profile: "cuda", "smemcuda", or "thrust"
    method_to_profile = "thrust"
    args = [
        "-i", input_file,
        "-k", "16",
        "-d", dims,
        "-e", method_to_profile,
        "-t", "0.0001",
        "-m", "50", # Use fewer iterations for a quicker profile
        "-s", "8675309",
    ]

    # Construct the nsys command
    output_filename = f"kmeans_profile_{precision}_{method_to_profile}"
    nsys_command = [
        "nsys", "profile",
        "-t", "cuda",  # Trace CUDA API calls, kernels, and memory operations
        "-o", output_filename,
        "--force-overwrite", "true",
        executable_path
    ] + args

    print(f"Executing: {' '.join(nsys_command)}\n")

    try:
        # Run the command and show output in real-time
        subprocess.run(nsys_command, check=True)
        report_file = f"{output_filename}.nsys-rep"
        print(f"\n--- Profiling Complete. Report file generated: {report_file} ---")
        print(f"To view the report, you can use the Nsight Systems UI or run: nsys stats {output_filename}.nsys-rep")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n>>> Profiling Failed! <<<")
        print(f"Error: {e}")
        print("Is 'nsys' in your system's PATH? It is part of the NVIDIA HPC SDK or CUDA Toolkit.")
        sys.exit(1)

def run_executable(precision="double"):
    """Runs the compiled k-means executable with example arguments."""
    print("\n--- Running K-Means Executable ---")
    
    executable_path = f"./bin/kmeans_{precision}" if precision == "float" else "./bin/kmeans"
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at '{executable_path}'. Did you run 'make'?")
        sys.exit(1)

    # --- K-MEANS ARGUMENTS ---
    input_file = "inputs/random-n2048-d16-c16.txt"
    dims = "16"
    # input_file = "inputs/random-n16384-d24-c16.txt"
    # dims = "24"
    input_file = "inputs/random-n65536-d32-c16.txt"
    dims = "32"
    args = [
        "-i", input_file,
        "-k", "16",
        "-d", dims,
        "-e", "smemcuda",
        "-t", "0.0001",
        "-m", "200",
        "-s", "8675309",
        # "-c" # Add for validation, remove for performance averaging.
    ]

    # Extract tolerance for Python validation from the '-t' flag.
    tolerance = 1e-5 # Default tolerance if -t is not in args
    try:
        t_index = args.index("-t")
        if t_index + 1 < len(args):
            tolerance = float(args[t_index + 1])
    except (ValueError, IndexError):
        print("Warning: '-t' flag not found in args list. Using default tolerance for Python validation.")

    command = [executable_path] + args
    print(f"Executing: {' '.join(command)}\n")

    # Run for performance if not validating, otherwise run once for validation.
    if "-c" not in args:
        run_and_average_performance(command, num_runs=20)
        return

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        final_centroids = parse_centroids(result.stdout)
        
        if final_centroids:
            answer_file = input_file.replace(".txt", "-answer.txt").replace("inputs/", "answers/")
            answer_centroids = read_points_file(answer_file, has_header=False)
            
            if answer_centroids:
                print(f"\n--- Comparing Final Centroids against Answer File: {answer_file} ---")
                compare_centroids(final_centroids, answer_centroids, tolerance=tolerance)
                points = read_points_file(input_file, has_header=True)
                validate_point_assignments(points, final_centroids, answer_centroids, tolerance=tolerance)

    except subprocess.CalledProcessError as e:
        print("\n>>> Execution Failed! <<<")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Command-line argument parsing
    # `python3 run.py` -> runs double
    # `python3 run.py float` -> runs float
    # `python3 run.py profile` -> profiles double
    # `python3 run.py profile float` -> profiles float
    
    mode = "run"
    target_precision = "double"

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "profile":
            mode = "profile"
            if len(sys.argv) > 2 and sys.argv[2].lower() == "float":
                target_precision = "float"
        elif sys.argv[1].lower() == "float":
            target_precision = "float"

    build_project()

    if mode == "profile":
        profile_executable(precision=target_precision)
    else:
        run_executable(precision=target_precision)

    print("\n--- Script Finished Successfully ---")