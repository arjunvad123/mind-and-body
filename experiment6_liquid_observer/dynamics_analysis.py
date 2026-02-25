"""
Dynamics Analysis Utilities

Functions for analyzing the dynamical properties of liquid neural networks:
- Time constant estimation
- Correlation dimension (Grassberger-Procaccia)
- Lyapunov exponents (Rosenstein 1993)
- Phase coherence (Hilbert transform)
- Transfer entropy
- Approximate integrated information (Phi)
- PCA reduction and trajectory clustering
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def approximate_tau(hidden_states, dt=0.02):
    """Approximate per-neuron time constants from hidden state trajectories.

    Since CfC doesn't expose tau directly, approximate from dynamics:
    tau_i(t) ~ -dt * h_i(t) / (h_i(t+1) - h_i(t))

    Args:
        hidden_states: (seq_len, hidden_size) numpy array
        dt: integration timestep

    Returns:
        tau: (seq_len-1, hidden_size) approximate time constants
    """
    dh = np.diff(hidden_states, axis=0)  # (seq_len-1, hidden_size)
    h = hidden_states[:-1]               # (seq_len-1, hidden_size)

    # Avoid division by near-zero
    mask = np.abs(dh) > 1e-6
    tau = np.full_like(dh, np.nan)

    tau[mask] = -dt * h[mask] / dh[mask]
    tau = np.clip(tau, 0.001, 1000.0)

    # Replace NaN with median
    for j in range(tau.shape[1]):
        col = tau[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) > 0:
            col[~np.isfinite(col)] = np.median(valid)
        else:
            col[:] = 1.0

    return tau


def compute_correlation_dimension(trajectory, max_dim=10, n_points=500):
    """Grassberger-Procaccia algorithm for estimating correlation dimension.

    Args:
        trajectory: (seq_len, dim) numpy array
        max_dim: maximum embedding dimension to try
        n_points: subsample size for efficiency

    Returns:
        estimated dimension (float)
    """
    # Subsample for efficiency
    if len(trajectory) > n_points:
        idx = np.random.choice(len(trajectory), n_points, replace=False)
        trajectory = trajectory[idx]

    # Compute pairwise distances
    dists = pdist(trajectory)
    if len(dists) == 0:
        return 0.0

    dists = dists[dists > 0]
    if len(dists) == 0:
        return 0.0

    # Correlation integral at multiple radii
    log_dists = np.log(dists)
    radii = np.logspace(np.percentile(log_dists, 5),
                        np.percentile(log_dists, 95),
                        num=20, base=np.e)

    log_r = []
    log_c = []
    n_pairs = len(dists)

    for r in radii:
        count = np.sum(dists < r)
        if count > 0:
            log_r.append(np.log(r))
            log_c.append(np.log(count / n_pairs))

    if len(log_r) < 3:
        return 0.0

    # Linear regression on scaling region
    log_r = np.array(log_r)
    log_c = np.array(log_c)

    # Use middle 60% as scaling region
    n = len(log_r)
    start = n // 5
    end = 4 * n // 5
    if end - start < 3:
        start, end = 0, n

    slope, _, _, _, _ = stats.linregress(log_r[start:end], log_c[start:end])

    return float(slope)


def compute_lyapunov_rosenstein(trajectory, dt=0.02, min_tsep=20, max_iter=None):
    """Largest Lyapunov exponent via Rosenstein (1993) nearest-neighbor method.

    Args:
        trajectory: (seq_len, dim) numpy array
        dt: time step
        min_tsep: minimum temporal separation for nearest neighbor search
        max_iter: maximum iterations for divergence tracking

    Returns:
        lambda_max: largest Lyapunov exponent (float)
    """
    n = len(trajectory)
    max_iter = max_iter or n // 4

    # Find nearest neighbors (not too close in time)
    dists = squareform(pdist(trajectory))

    # Mask out temporally close points
    for i in range(n):
        for j in range(max(0, i - min_tsep), min(n, i + min_tsep + 1)):
            dists[i, j] = np.inf

    # For each point, find its nearest neighbor
    nn_idx = np.argmin(dists, axis=1)
    nn_dist = dists[np.arange(n), nn_idx]

    # Track divergence
    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)

    for i in range(n):
        j = nn_idx[i]
        d0 = nn_dist[i]
        if d0 < 1e-10 or not np.isfinite(d0):
            continue

        for k in range(max_iter):
            if i + k >= n or j + k >= n:
                break
            dist = np.linalg.norm(trajectory[i + k] - trajectory[j + k])
            if dist > 0:
                divergence[k] += np.log(dist / d0)
                counts[k] += 1

    # Average divergence
    valid = counts > 0
    if not np.any(valid):
        return 0.0

    avg_divergence = np.zeros(max_iter)
    avg_divergence[valid] = divergence[valid] / counts[valid]

    # Fit slope to linear region
    time_axis = np.arange(max_iter) * dt
    valid_idx = np.where(valid)[0]

    if len(valid_idx) < 3:
        return 0.0

    # Use first quarter as most linear
    end = max(3, len(valid_idx) // 4)
    idx = valid_idx[:end]

    slope, _, _, _, _ = stats.linregress(time_axis[idx], avg_divergence[idx])

    return float(slope)


def compute_phase_coherence(signal1, signal2):
    """Mean phase coherence via Hilbert transform.

    Args:
        signal1: (seq_len,) numpy array
        signal2: (seq_len,) numpy array

    Returns:
        coherence: float in [0, 1]. 1 = perfectly synchronized.
    """
    # Analytic signal via Hilbert transform
    analytic1 = scipy_signal.hilbert(signal1)
    analytic2 = scipy_signal.hilbert(signal2)

    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)

    # Phase difference
    phase_diff = phase1 - phase2

    # Mean phase coherence (Kuramoto order parameter)
    coherence = np.abs(np.mean(np.exp(1j * phase_diff)))

    return float(coherence)


def compute_transfer_entropy(source, target, lag=1, bins=10):
    """Transfer entropy from source to target at given lag.

    TE(X->Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})

    Uses binned estimation.

    Args:
        source: (seq_len,) numpy array
        target: (seq_len,) numpy array
        lag: time lag
        bins: number of bins for discretization

    Returns:
        transfer_entropy: float (bits)
    """
    n = min(len(source), len(target))
    if n < lag + 2:
        return 0.0

    # Discretize
    src_bins = np.digitize(source[:n], np.linspace(source.min(), source.max(), bins + 1)[:-1])
    tgt_bins = np.digitize(target[:n], np.linspace(target.min(), target.max(), bins + 1)[:-1])

    # Align: Y_t, Y_{t-1}, X_{t-lag}
    y_t = tgt_bins[lag:]
    y_prev = tgt_bins[lag - 1:-1] if lag > 0 else tgt_bins[:-1]
    x_lag = src_bins[:len(y_t)]

    # Joint and marginal distributions
    n_samples = len(y_t)
    if n_samples < 10:
        return 0.0

    # H(Y_t | Y_{t-1}) via joint entropy
    def entropy_cond(a, b):
        """H(A|B) = H(A,B) - H(B)"""
        joint = np.column_stack([a, b])
        _, counts_joint = np.unique(joint, axis=0, return_counts=True)
        _, counts_b = np.unique(b, return_counts=True)

        p_joint = counts_joint / n_samples
        p_b = counts_b / n_samples

        h_joint = -np.sum(p_joint * np.log2(p_joint + 1e-10))
        h_b = -np.sum(p_b * np.log2(p_b + 1e-10))

        return h_joint - h_b

    h_y_given_yprev = entropy_cond(y_t, y_prev)
    h_y_given_yprev_x = entropy_cond(y_t, np.column_stack([y_prev, x_lag]))

    te = h_y_given_yprev - h_y_given_yprev_x
    return float(max(0, te))


def compute_gaussian_phi(states_t, states_t1, n_partitions=20):
    """Approximate Phi via Gaussian mutual information over random bipartitions.

    Method:
    1. Estimate joint covariance of (X_t, X_{t+1})
    2. For random bipartitions (A, B):
       Phi_partition = MI(whole) - sum(MI(parts))
    3. Phi = min over partitions (MIP)

    Args:
        states_t: (n_samples, n_neurons) — states at time t
        states_t1: (n_samples, n_neurons) — states at time t+1
        n_partitions: number of random bipartitions to try

    Returns:
        phi_approx: float
    """
    n_samples, n_neurons = states_t.shape

    if n_samples < n_neurons * 2:
        return 0.0

    def gaussian_mi(x, y):
        """Mutual information assuming Gaussian distributions."""
        n = len(x)
        if n < 5 or x.shape[1] == 0 or y.shape[1] == 0:
            return 0.0

        # Covariance matrices
        cov_x = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
        cov_y = np.cov(y, rowvar=False) + np.eye(y.shape[1]) * 1e-6
        joint = np.column_stack([x, y])
        cov_joint = np.cov(joint, rowvar=False) + np.eye(joint.shape[1]) * 1e-6

        # MI = 0.5 * log(det(Cx) * det(Cy) / det(C_joint))
        sign_x, logdet_x = np.linalg.slogdet(cov_x)
        sign_y, logdet_y = np.linalg.slogdet(cov_y)
        sign_j, logdet_j = np.linalg.slogdet(cov_joint)

        if sign_x <= 0 or sign_y <= 0 or sign_j <= 0:
            return 0.0

        mi = 0.5 * (logdet_x + logdet_y - logdet_j)
        return float(max(0, mi))

    # Whole system MI
    whole_mi = gaussian_mi(states_t, states_t1)

    # Random bipartitions
    rng = np.random.RandomState(42)
    phi_values = []

    for _ in range(n_partitions):
        # Random bipartition of neurons
        perm = rng.permutation(n_neurons)
        split = max(1, n_neurons // 2)
        part_a = perm[:split]
        part_b = perm[split:]

        if len(part_a) == 0 or len(part_b) == 0:
            continue

        # MI of each part with its own future
        mi_a = gaussian_mi(states_t[:, part_a], states_t1[:, part_a])
        mi_b = gaussian_mi(states_t[:, part_b], states_t1[:, part_b])

        # Also cross-partition MI
        mi_a_to_b = gaussian_mi(states_t[:, part_a], states_t1[:, part_b])
        mi_b_to_a = gaussian_mi(states_t[:, part_b], states_t1[:, part_a])

        parts_mi = mi_a + mi_b
        phi_partition = whole_mi - parts_mi

        phi_values.append(phi_partition)

    if not phi_values:
        return 0.0

    # Phi = minimum information partition (MIP)
    return float(min(phi_values))


def pca_reduce(trajectories, n_components=3):
    """PCA reduction for visualization.

    Args:
        trajectories: (n_traj, seq_len, dim) or (seq_len, dim)

    Returns:
        reduced: same shape with dim replaced by n_components
        pca: fitted PCA object
        variance_explained: array of explained variance ratios
    """
    original_shape = trajectories.shape
    is_3d = len(original_shape) == 3

    if is_3d:
        flat = trajectories.reshape(-1, original_shape[-1])
    else:
        flat = trajectories

    pca = PCA(n_components=n_components)
    reduced_flat = pca.fit_transform(flat)

    if is_3d:
        reduced = reduced_flat.reshape(original_shape[0], original_shape[1], n_components)
    else:
        reduced = reduced_flat

    return reduced, pca, pca.explained_variance_ratio_


def cluster_trajectories(trajectories, system_labels):
    """Silhouette score for trajectory clustering by system type.

    Args:
        trajectories: (n_traj, seq_len, dim) — reduce to feature vectors first
        system_labels: (n_traj,) string labels

    Returns:
        silhouette: float in [-1, 1]. Higher = better separation.
    """
    # Feature: mean trajectory (average over time)
    features = trajectories.mean(axis=1)  # (n_traj, dim)

    # Encode labels to integers
    unique_labels = np.unique(system_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([label_map[l] for l in system_labels])

    if len(unique_labels) < 2:
        return 0.0

    try:
        score = silhouette_score(features, int_labels)
    except ValueError:
        score = 0.0

    return float(score)
