import numpy as np
import sys
import os
from scipy.optimize import linear_sum_assignment



def read_points(filename):
    """Reads points from a text file into a NumPy array."""
    try:
        with open(filename, 'r') as f:
            # Skip the header line which contains the point count
            f.readline()
            # Read all subsequent lines
            lines = f.readlines()
            # Process each line, skipping the initial index
            points = [list(map(float, line.strip().split()[1:])) for line in lines if line.strip()]
        return np.array(points, dtype=np.float64)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filename}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading or parsing file '{filename}': {e}", file=sys.stderr)
        return None

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
                # Files have a leading index for each point/centroid.
                # We split the line and take all but the first element.
                point = [float(n) for n in line.split()[1:]]
                points.append(point)
        except (IOError, ValueError) as e:
            print(f"Error reading point file '{filepath}': {e}")
    return points

def initialize_centroids(points, k, seed):
    """
    Initializes centroids by randomly selecting k points from the dataset.
    This mimics the simple (non-unique) random sampling from the C++ implementation.
    """
    np.random.seed(seed)
    num_points = points.shape[0]
    
    # Generate k random indices, allowing for duplicates
    random_indices = np.random.randint(0, num_points, size=k)
    
    # Select the points corresponding to the random indices
    centroids = points[random_indices].copy()
    return centroids

def assign_clusters(points, centroids):
    """
    Assigns each point to the nearest centroid using squared Euclidean distance.
    
    Args:
        points (np.array): Shape (num_points, dims)
        centroids (np.array): Shape (k, dims)
        
    Returns:
        np.array: Shape (num_points,) with the index of the assigned cluster for each point.
    """
    # Use broadcasting to calculate the difference between each point and every centroid
    # points shape: (N, 1, D), centroids shape: (1, K, D) -> diff shape: (N, K, D)
    diff = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    
    # Square the differences and sum along the dimension axis to get squared Euclidean distance
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Find the index of the minimum distance for each point
    assignments = np.argmin(dist_sq, axis=1)
    return assignments

def update_centroids(points, assignments, k):
    """
    Recalculates the centroids as the mean of all points assigned to each cluster.
    
    Args:
        points (np.array): The dataset.
        assignments (np.array): The cluster assignment for each point.
        k (int): The number of clusters.
        
    Returns:
        np.array: The new centroids.
    """
    dims = points.shape[1]
    new_centroids = np.zeros((k, dims), dtype=np.float64)
    
    for i in range(k):
        # Find all points assigned to the current cluster
        points_in_cluster = points[assignments == i]
        
        # If the cluster is not empty, calculate the mean
        if len(points_in_cluster) > 0:
            new_centroids[i] = np.mean(points_in_cluster, axis=0)
            
    return new_centroids

def python_kmeans(points, k, max_iter, threshold, seed):
    """
    Performs k-means clustering using Python and NumPy.
    """
    print("--- Executing Python K-Means ---")
    
    # 1. Initialize centroids
    centroids = initialize_centroids(points, k, seed)
    
    for i in range(max_iter):
        old_centroids = centroids.copy()
        
        # 2. Assignment Step
        assignments = assign_clusters(points, centroids)
        
        # 3. Update Step
        centroids = update_centroids(points, assignments, k)
        
        # 4. Convergence Check
        # Calculate the squared distance each centroid has moved
        diff = np.sum((centroids - old_centroids) ** 2, axis=1)
        
        # If the maximum movement is less than the threshold, we have converged
        if np.max(diff) < threshold**2:
            print(f"Convergence reached after {i + 1} iterations.")
            break
    else: # This 'else' belongs to the 'for' loop, runs if the loop completes without 'break'
        print(f"Max iterations ({max_iter}) reached without convergence.")
        
    return centroids

def print_centroid_list(name, centroids):
    """Helper function to print a list of centroids with formatting."""
    print(f"\n--- {name} (CSV Format) ---")
    for i, centroid in enumerate(centroids):
        # Create a list of strings for all coordinates, formatted to 10 decimal places.
        coords_str_list = [f"{x:.10f}" for x in centroid]
        # Join the index and all coordinates with a comma to create a CSV line.
        csv_line = ",".join([str(i)] + coords_str_list)
        print(csv_line)

def compare_centroids(calculated, answers, tolerance=1e-5):
    """
    Compares two sets of centroids for similarity using the Hungarian algorithm
    to find the optimal one-to-one mapping.
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
 
    # Print both sets of centroids for inspection.
    print_centroid_list("Final Centroids (Python Calculated)", calculated)
    print_centroid_list("Answer Centroids (Ground Truth)", answers)

    # --- Display the distance for each matched pair ---
    print("\n--- Distances between Matched Python and Answer Centroids ---")
    distances = cost_matrix[row_ind, col_ind]
    max_distance = distances.max()
 
    if max_distance > tolerance:
        print(f"\nValidation Failed: Maximum centroid distance ({max_distance:.6f}) exceeds tolerance ({tolerance}).")
        return False
 
    print(f"\nValidation Successful! All matched centroid pairs are within tolerance. Max distance: {max_distance:.6f}")
    return True

if __name__ == '__main__':
    # Example usage, mirroring the arguments from run.py
    input_file = "inputs/random-n2048-d16-c16.txt"
    k = 16
    dims = 16
    threshold = 0.00001
    max_iter = 150
    seed = 20
    
    points = read_points(input_file)
    
    if points is not None:
        final_centroids = python_kmeans(points, k, max_iter, threshold, seed)
        
        # --- Load answer centroids for validation ---
        answer_file = input_file.replace(".txt", "-answer.txt").replace("inputs/", "answers/")
        answer_centroids = read_points_file(answer_file, has_header=False)
        
        if answer_centroids:
            print(f"\n--- Comparing Python Centroids against Answer File: {answer_file} ---")
            compare_centroids(final_centroids, answer_centroids)