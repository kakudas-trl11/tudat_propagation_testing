import requests
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt


def query_starlink_ephemeris_file(starlink_ephemeris_file: str):
    STARLINK_EPHERIS_URL = f"https://api.starlink.com/public-files/ephemerides/{starlink_ephemeris_file}"
    
    ephemeris_entries = []
    with requests.get(STARLINK_EPHERIS_URL, stream=True) as response:
        group = []
        for i, line in enumerate(response.iter_lines(decode_unicode=True)):
            # print(f"[DEBUG] Line {i}: {line}")
            group.append(line)
            if len(group) == 4:
                ephemeris_entries.append(group)
                group = []
            # Break after a certain number of groups
            if len(ephemeris_entries) > 2:
                break
    return ephemeris_entries


def parse_starlink_ephemeris(filename):
    with open(filename) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 7:
                # Parse state vector
                time_str = parts[0]
                state = np.array([float(x) for x in parts[1:7]])
            elif line.startswith("cov:"):
                # Parse covariance
                cov_flat = [float(x) for x in line.replace("cov:", "").split()]
                covariance = np.array(cov_flat).reshape((6, 6))
    return time_str, state, covariance


def parse_starlink_ephemeris_lines(entry_lines: List[str]):
    state_vector = []
    covariance_matrix = []
    time_str = ""
    for i, line in enumerate(entry_lines):
        if i == 0:
            parts = line.split()
            time_str = parts[0]
            state_vector = [float(x) for x in parts[1:7]]
        elif 1 <= i <= 3:
            cov_line = [float(x) for x in line.split()]
            covariance_matrix.extend(cov_line)
    return time_str, state_vector, covariance_matrix


def yyyydddhhmmss_to_components(timestr):
    year = int(timestr[:4])
    doy = int(timestr[4:7])
    hour = int(timestr[7:9])
    minute = int(timestr[9:11])
    second = float(timestr[11:])
    # Convert day-of-year to month and day
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    month = dt.month
    day = dt.day
    return year, month, day, hour, minute, second


def yyyydddhhmmss_to_datetime(timestr):
    from datetime import datetime, timedelta
    year = int(timestr[:4])
    doy = int(timestr[4:7])
    hour = int(timestr[7:9])
    minute = int(timestr[9:11])
    second = float(timestr[11:])
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second)
    return dt


def covariance_from_lower_triangular(vec21):
    P = np.zeros((6,6))
    idx = 0
    for i in range(6):
        for j in range(i+1):
            P[i,j] = vec21[idx]
            P[j,i] = vec21[idx]  # symmetry
            idx += 1
    return P


from scipy.special import i0


def foster_pc(P2D, R, n_theta=2000):
    theta = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)

    sigma2 = (
        P2D[0,0]*c**2 +
        2*P2D[0,1]*c*s +
        P2D[1,1]*s**2
    )

    if np.any(sigma2 <= 0):
        raise ValueError("Covariance is not positive definite")

    integrand = 1.0 - np.exp(-R**2 / (2.0 * sigma2))
    Pc = np.trapz(integrand, theta) / (2*np.pi)

    return Pc


def compute_collision_probability(relative_position, combined_covariance, threshold_distance):
    """
    Compute the Mahalanobis distance and collision probability for two satellites.
    
    Parameters:
    -----------
    relative_position : array-like, shape (3,)
        The relative position vector between two satellites (r2 - r1) in meters
    combined_covariance : array-like, shape (3, 3)
        The combined covariance matrix (P1 + P2) in meters^2
    threshold_distance : float
        The collision threshold distance in meters (e.g., sum of satellite radii + safety margin)
    
    Returns:
    --------
    dict with keys:
        'mahalanobis_distance': float
            The Mahalanobis distance (unitless)
        'miss_distance': float
            The Euclidean distance between satellites in meters
        'probability_of_collision': float
            Approximate probability of collision (0 to 1)
        'sigma_level': float
            How many standard deviations apart the satellites are
    """
    # Convert inputs to numpy arrays
    r = np.array(relative_position).reshape(3, 1)
    P = np.array(combined_covariance)
    
    # Compute miss distance (Euclidean distance)
    miss_distance = np.linalg.norm(r)
    
    # Compute Mahalanobis distance
    # d^2 = r^T * P^(-1) * r
    P_inv = np.linalg.inv(P)
    mahalanobis_squared = (r.T @ P_inv @ r).item()
    mahalanobis_distance = np.sqrt(mahalanobis_squared)
    
    # The Mahalanobis distance squared follows a chi-squared distribution with 3 DOF
    # This gives us the probability that the true relative position is within
    # the uncertainty ellipsoid
    p_within_ellipsoid = stats.chi2.cdf(mahalanobis_squared, df=3)
    
    # For collision probability, we use a 2D approximation (Foster-like approach)
    # This projects the 3D problem onto the 2D collision plane
    # First, find the projection of relative position onto the plane perpendicular to velocity
    # For simplicity, we'll use a circular cross-section approximation
    
    # Compute the probability that satellites are within threshold distance
    # This is an approximation based on the relative position uncertainty
    
    # Probability of collision calculation
    # We need P(||r_true|| < threshold) where r_true ~ N(r_mean, P)
    # This is the probability that the true relative position is within the collision sphere
    
    # Monte Carlo integration approach for accurate probability
    # Generate samples from the 3D Gaussian distribution
    n_samples = 100000
    mean = r.flatten()
    samples = np.random.multivariate_normal(mean, P, n_samples)
    
    # Compute distance for each sample
    distances = np.linalg.norm(samples, axis=1)
    
    # Probability is fraction of samples within threshold
    pc_monte_carlo = np.sum(distances < threshold_distance) / n_samples
     
    # For each point on the threshold sphere, compute its Mahalanobis distance
    # This is complex, so we'll use the Monte Carlo result
    pc_approx = pc_monte_carlo
    
    return {
        'mahalanobis_distance': mahalanobis_distance,
        'miss_distance': miss_distance,
        'probability_of_collision': pc_approx,
        'sigma_level': mahalanobis_distance,
        'threshold_distance': threshold_distance
    }


def main():
    # Query Starlink api
    STARLINK_MANIFEST_URL = "https://api.starlink.com/public-files/ephemerides/MANIFEST.txt"

    starlink_ephemeris_files = []
    with requests.get(STARLINK_MANIFEST_URL) as response:
        response.raise_for_status()
        for i, line in enumerate(response.text.splitlines()):
            if len(line) > 0:
                starlink_ephemeris_file = line
                starlink_ephemeris_files.append(starlink_ephemeris_file)
            if i >= 4:
                break
    
    ephemeris_file_a = starlink_ephemeris_files[0]
    ephemeris_file_b = starlink_ephemeris_files[0] # (use same satellite but different epochs)
    print(f"[DEBUG] Using ephemeris files:\n A: {ephemeris_file_a}\n B: {ephemeris_file_b}")

    # Extract ephemeris data
    parts_a = ephemeris_file_a.split("_")
    NORAD_ID_a = parts_a[1]
    STARLINK_ID_a = parts_a[2]

    parts_b = ephemeris_file_b.split("_")
    NORAD_ID_b = parts_b[1]
    STARLINK_ID_b = parts_b[2]

    ephemeris_entries_a = query_starlink_ephemeris_file(ephemeris_file_a)
    ephemeris_entries_b = query_starlink_ephemeris_file(ephemeris_file_b)
    print(f"[DEBUG] Retrieved {len(ephemeris_entries_a)} entries for Starlink {STARLINK_ID_a}")
    print(f"[DEBUG] Retrieved {len(ephemeris_entries_b)} entries for Starlink {STARLINK_ID_b}")
    
    # Extract last entry for each satellite
    time_str_a, state_vector_a, covariance_matrix_a = parse_starlink_ephemeris_lines(ephemeris_entries_a[-1])
    time_str_b, state_vector_b, covariance_matrix_b = parse_starlink_ephemeris_lines(ephemeris_entries_b[-2])

    print(f"[DEBUG] Starlink {STARLINK_ID_a} state vector: {state_vector_a}")
    print(f"[DEBUG] Starlink {STARLINK_ID_b} state vector: {state_vector_b}")
    cov_matrix_6x6_a = covariance_from_lower_triangular(covariance_matrix_a) * 1E10
    cov_matrix_6x6_b = covariance_from_lower_triangular(covariance_matrix_b) * 1E10
    
    # Perform conjunction analysis
    pos_rel = np.array(state_vector_b[:3]) - np.array(state_vector_a[:3])
    cov_rel = cov_matrix_6x6_a[0:3, 0:3] + cov_matrix_6x6_b[0:3, 0:3]
    cov_rel = 0.5 * (cov_rel + cov_rel.T)

    print("[DEBUG] Relative Position Vector (km):", pos_rel)
    print("[DEBUG] Relative Covariance Matrix (km^2):\n", cov_rel)

    eigvals, eigvecs = np.linalg.eigh(cov_rel)
    eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
    cov_rel_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    
    min_R = 100
    max_R = 1000
    num_R = 100
    R_values = np.linspace(min_R, max_R, num_R)

    rel_pos_unit = pos_rel / np.linalg.norm(pos_rel)
    # Projection matrix to the plane normal to rel_pos_unit
    P = np.eye(3) - np.outer(rel_pos_unit, rel_pos_unit)
    # 2D projected covariance
    P2D = P @ cov_rel_psd @ P
    # Remove the zero eigenvalue direction (project to 2D)
    eigvals2d, eigvecs2d = np.linalg.eigh(P2D)
    idx = np.argsort(eigvals2d)[::-1][:2]
    eigvecs2d = eigvecs2d[:, idx]
    P2D_final = eigvecs2d.T @ P2D @ eigvecs2d

    # Use the Foster function for Pc as a function of R
    mu = np.linalg.norm(pos_rel)
    Pc_values = []

    for R in R_values:
        result = compute_collision_probability(pos_rel, cov_rel_psd, R)
        Pc_values.append(result['probability_of_collision'])

    # Plot probability of collision vs. distance using the approximation
    plt.figure(figsize=(8, 5))
    plt.plot(R_values, Pc_values, marker='o')
    plt.xlabel("Conjunction Distance Threshold (km)")
    plt.ylabel("Probability of Collision (Pc)")
    plt.title(f"Pc vs. Distance (Approximate) for Starlink {STARLINK_ID_a} and {STARLINK_ID_b}")
    plt.grid()
    plt.tight_layout()
    plt.savefig("starlink_conjunction_pc_vs_distance_approx.png", dpi=200)
    plt.close()



if __name__  == "__main__":
    main()


# TODO Implement probability mass within threshold sphere
# Using the chi-squared distribution for radial distance
# The radial distance squared (in Mahalanobis space) follows chi2(3)
# Transform threshold distance to Mahalanobis space
# For a sphere of radius R, find equivalent chi-squared threshold
if miss_distance > 0:
    # The threshold creates a sphere; we need to find the chi-squared value
    # that corresponds to this sphere in the warped Mahalanobis space
    threshold_mahalanobis_sq = threshold_distance**2 / np.mean(eigenvalues)
else:
    threshold_mahalanobis_sq = threshold_distance**2 / np.mean(eigenvalues)