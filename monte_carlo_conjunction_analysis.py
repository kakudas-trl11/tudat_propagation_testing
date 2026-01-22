import numpy as np
from math import exp, factorial
from scipy.stats import chi2, ncx2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def random_covariance_matrix(dim=3):
    """Generate a random positive definite covariance matrix."""
    A = np.random.randn(dim, dim)
    return np.dot(A, A.T) + np.eye(dim) * 0.1


def ellipsoid_surface_points(mean, cov, confidence=0.95, n_u=80, n_v=40):
    """Sample points on a 3D confidence ellipsoid surface.

    mean: (3,)
    cov: (3,3)
    Returns:
      X, Y, Z: (n_v, n_u) grids for plotting
      pts: (n_v*n_u, 3) point cloud on the surface
    """
    mean = np.asarray(mean, dtype=float).reshape(3)
    cov = np.asarray(cov, dtype=float).reshape(3, 3)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    chi2_val = chi2.ppf(confidence, 3)
    radii = np.sqrt(np.maximum(vals, 0.0) * chi2_val)

    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)

    # Unit sphere parameterization
    xs = np.outer(np.ones_like(v), cu) * np.outer(sv, np.ones_like(u))
    ys = np.outer(np.ones_like(v), su) * np.outer(sv, np.ones_like(u))
    zs = np.outer(cv, np.ones_like(u))

    # Scale to ellipsoid in principal-axis frame
    x0 = radii[0] * xs
    y0 = radii[1] * ys
    z0 = radii[2] * zs

    # Rotate + translate
    pts0 = np.stack([x0.ravel(), y0.ravel(), z0.ravel()], axis=0)  # (3, N)
    pts = (vecs @ pts0).T + mean[None, :]  # (N, 3)

    X = pts[:, 0].reshape(n_v, n_u)
    Y = pts[:, 1].reshape(n_v, n_u)
    Z = pts[:, 2].reshape(n_v, n_u)
    return X, Y, Z, pts


def chan_probability_within_distance(
    mean1: np.ndarray,
    cov1: np.ndarray,
    mean2: np.ndarray,
    cov2: np.ndarray,
    threshold_distance: float
) -> float:
    """
    Chan's approximation for probability that two 3D Gaussian-distributed
    points are within a given Euclidean distance.

    Parameters
    ----------
    mean1, mean2 : (3,) ndarray
        Mean position vectors
    cov1, cov2 : (3,3) ndarray
        Covariance matrices
    threshold_distance : float
        Distance threshold

    Returns
    -------
    prob : float
        P(||x1 - x2|| <= threshold_distance)
    """

    # --- Basic validation ---
    mean1 = np.asarray(mean1, dtype=float)
    mean2 = np.asarray(mean2, dtype=float)
    cov1 = np.asarray(cov1, dtype=float)
    cov2 = np.asarray(cov2, dtype=float)

    if mean1.shape != (3,) or mean2.shape != (3,):
        raise ValueError("Means must be 3D vectors")

    if cov1.shape != (3, 3) or cov2.shape != (3, 3):
        raise ValueError("Covariances must be 3x3 matrices")

    if threshold_distance <= 0.0:
        return 0.0

    # --- Relative mean and covariance ---
    mu_r = mean1 - mean2                  # relative mean
    P_r = cov1 + cov2                     # relative covariance

    # --- Moments of Z = ||r||^2 ---
    mu_norm_sq = mu_r @ mu_r

    trace_P = np.trace(P_r)
    trace_P2 = np.trace(P_r @ P_r)

    mean_Z = trace_P + mu_norm_sq
    var_Z = 2.0 * trace_P2 + 4.0 * (mu_r @ P_r @ mu_r)

    if mean_Z <= 0.0 or var_Z <= 0.0:
        return 0.0

    # --- Chan scaling ---
    alpha = var_Z / (2.0 * mean_Z)

    if alpha <= 0.0:
        return 0.0

    # Noncentrality parameter
    lambda_ = mu_norm_sq / alpha

    # Effective degrees of freedom
    k = 2.0 * mean_Z * mean_Z / var_Z

    if k <= 0.0:
        return 0.0

    # --- Evaluate noncentral chi-square CDF ---
    x = (threshold_distance ** 2) / alpha

    return ncx2.cdf(x, df=k, nc=lambda_)


def alfano_probability_within_distance(
    mean1: np.ndarray,
    cov1: np.ndarray,
    mean2: np.ndarray,
    cov2: np.ndarray,
    threshold_distance: float,
    max_terms: int = 200,
    tol: float = 1e-12
) -> float:
    """
    Alfano exact probability for P(||x1 - x2|| <= d),
    where x1, x2 are 3D Gaussians.

    Deterministic, no Monte Carlo.
    """

    # --- Validation ---
    mean1 = np.asarray(mean1, dtype=float)
    mean2 = np.asarray(mean2, dtype=float)
    cov1 = np.asarray(cov1, dtype=float)
    cov2 = np.asarray(cov2, dtype=float)

    if mean1.shape != (3,) or mean2.shape != (3,):
        raise ValueError("Means must be 3D vectors")
    if cov1.shape != (3, 3) or cov2.shape != (3, 3):
        raise ValueError("Covariances must be 3x3 matrices")
    if threshold_distance <= 0.0:
        return 0.0

    # --- Relative statistics ---
    mu = mean1 - mean2
    P = cov1 + cov2

    # --- Eigen-decomposition ---
    eigvals, eigvecs = np.linalg.eigh(P)

    # Guard against numerical negatives
    eigvals = np.maximum(eigvals, 0.0)

    # Transform mean
    mu_t = eigvecs.T @ mu

    # --- Noncentrality parameters ---
    deltas = (mu_t ** 2) / eigvals

    # --- Radius ---
    r2 = threshold_distance ** 2

    # --- Alfano series ---
    prob = 0.0
    weight_sum = 0.0

    for n in range(max_terms):
        # Poisson weight
        w = exp(-0.5 * np.sum(deltas)) * (0.5 * np.sum(deltas)) ** n / factorial(n)

        dof = 3 + 2 * n

        # Scaled chi-square argument
        x = r2 / np.mean(eigvals)

        term = w * chi2.cdf(x, dof)

        prob += term
        weight_sum += w

        # Convergence check
        if w < tol:
            break

    # Normalize in case of truncation
    return min(prob / weight_sum, 1.0)


def main():
    # Generate two random 3D points and their covariance matrices
    # np.random.seed(42)
    # mean1 = np.random.uniform(-5, 5, 3)
    # mean2 = np.random.uniform(-5, 5, 3)
    # cov1 = random_covariance_matrix(dim=3)
    # cov2 = random_covariance_matrix(dim=3)

    mean1 = [0, 5, 0]
    mean2 = [5, 0, 0]

    cov1 = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]    
    
    cov2 = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]] 


    threshold_distance = 10
    # threshold_distance = np.random.uniform(1, 10)
    print(f"Threshold distance: {threshold_distance}")

    # Chan's approximation
    chan_prob = chan_probability_within_distance(mean1, cov1, mean2, cov2, threshold_distance)
    print(f"Chan's approximation of probability within threshold: {chan_prob:.6f}")

    # Alfano's exact method
    alfano_prob = alfano_probability_within_distance(mean1, cov1, mean2, cov2, threshold_distance)
    print(f"Alfano's exact probability within threshold: {alfano_prob:.6f}")

    sample_ranges = [100, 500, 1000, 5000, 10000, 50000, 100000]

    for n_samples in sample_ranges:
        pts1 = np.random.multivariate_normal(mean1, cov1, n_samples)
        pts2 = np.random.multivariate_normal(mean2, cov2, n_samples)
        distances = np.linalg.norm(pts1 - pts2, axis=1)
        prob = np.mean(distances < threshold_distance)
        print(f"Num Samples: {n_samples:8d} | Prob within threshold: {prob}")

        # Visualize only for the first sample size
        if n_samples == sample_ranges[2]:
        # if False:
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], color='r', alpha=0.3, label='Ellipsoid 1 samples')
            ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], color='b', alpha=0.3, label='Ellipsoid 2 samples')
            ax.scatter(mean1[0], mean1[1], mean1[2], color='r', s=80, marker='x', label='Mean 1')
            ax.scatter(mean2[0], mean2[1], mean2[2], color='b', s=80, marker='x', label='Mean 2')

            # Optional: plot threshold sphere around mean1
            # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            # xs = mean1[0] + threshold_distance * np.cos(u) * np.sin(v)
            # ys = mean1[1] + threshold_distance * np.sin(u) * np.sin(v)
            # zs = mean1[2] + threshold_distance * np.cos(v)
            # ax.plot_wireframe(xs, ys, zs, color='g', alpha=0.2, label='Threshold sphere')
            X1, Y1, Z1, _ = ellipsoid_surface_points(mean1, cov1, confidence=0.95)
            X2, Y2, Z2, _ = ellipsoid_surface_points(mean2, cov2, confidence=0.95)
            ax.plot_wireframe(X1, Y1, Z1, color='r', alpha=0.5, linewidth=0.7, label='Ellipsoid 1')
            ax.plot_wireframe(X2, Y2, Z2, color='b', alpha=0.5, linewidth=0.7, label='Ellipsoid 2')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.set_title(f"Monte Carlo samples (n={n_samples})")
            plt.tight_layout()
            plt.show()
   
    


if __name__ == "__main__":
    main()