import numpy as np
from plotters import plot_prediction_part_2, plot_mse

"""
👆 Part (a)
"""

"""
Hyperparameters
"""

N = 10_000  # Number of trials
T = 100     # Number of timesteps
time = 10   # Number of seconds
dt = 0.1    # Size of each timestep

"""
Transition matrices
"""

A = np.array([
    [1, dt, 0, 0],
    [0, 1, dt, 0],
    [0, 0, 1, dt],
    [0, 0, 0, 1],
]) # state transition matrix

C = np.array([
    [1, 0, 0, 0]
]) # measurement probability matrix

"""
Gaussian noise
"""

vm = 0     # mean
Q = 1.0    # variance

"""
Initial conditions
"""

mu0 = np.array([[5, 1, 0, 0]])
Sigma0 = np.diag([10, 10, 10, 10])

def get_true_position(t: int) -> int:
    return np.sin(0.1 * t)

def get_gaussian_noise() -> int:
    return np.random.normal(vm, np.sqrt(Q))

def kalman_filter(mu, Sigma, z) -> tuple:
    """
    Updates mu, Sigma based on a new observation z
    Assumes no process noise
    """
    
    mu_pred = A @ mu
    Sigma_pred = A @ Sigma @ A.T

    K = Sigma_pred @ C.T @ np.linalg.inv(C @ Sigma_pred @ C.T + Q)
    mu_new = mu_pred + K @ (z - C @ mu_pred)
    Sigma_new = (np.eye(len(K)) - K @ C) @ Sigma_pred

    return mu_new, Sigma_new

def calc_prediction(filter_function) -> tuple:
    """
    For a single trial, calculate MSE from t = 1 ... 100

    Returns:
    - timesteps
    - ground truths
    - measurements
    - predicted means
    - predicted covariances
    """
    
    mu = mu0.T

    x_true = 0
    z_noisy = x_true + get_gaussian_noise()
    x0 = mu0[0][0]
    Sigma = Sigma0

    timesteps = np.arange(0, time, dt)
    ground_truths = [x_true]
    measurements = [z_noisy]
    predict_means = [x0]
    predict_cov = [Sigma0]

    for t in range(1, T):
        x_true = get_true_position(t)
        ground_truths.append(x_true)

        z_noisy = x_true + get_gaussian_noise()
        measurements.append(z_noisy)

        mu, Sigma = filter_function(mu, Sigma, z_noisy)
        x_new = mu[0][0]
        predict_means.append(x_new)
        predict_cov.append(Sigma)    
    
    ground_truths = np.array(ground_truths).reshape((-1, 1))
    measurements = np.array(measurements).reshape((-1, 1))
    predict_means = np.array(predict_means).reshape((-1, 1))
    predict_cov = np.stack(predict_cov)

    return timesteps, ground_truths, measurements, predict_means, predict_cov

def calc_mse(filter_function) -> int:
    """
    Over N = 10,000 trials, find the MSE

    Returns:
    - timesteps
    - ground truths
    - predicted means
    """
    
    timesteps = np.arange(0, time, dt)
    ground_truths = np.array([get_true_position(t) for t in np.arange(0, T)])
    predict_means_list = []

    for _ in range(N):
        _, _, _, predict_means, _ = calc_prediction(filter_function)
        predict_means_list.append(predict_means)

    return timesteps, ground_truths, np.stack(predict_means_list)

"""
Plot graphs
"""

prediction_metrics = calc_prediction(kalman_filter)
plot_prediction_part_2(*prediction_metrics)

mse_metrics = calc_mse(kalman_filter)
plot_mse(*mse_metrics)

"""
✌️ Part (b)
"""

"""
Fictitious state transition probability noise
"""

wm = 0
R = np.diag([0.1, 0.1, 0.1, 0.1])

def noisy_kalman_filter(mu, Sigma, z):
    """
    Updates mu, Sigma based on a new observation z, taking into account noise R
    """

    mu_pred = A @ mu
    Sigma_pred = A @ Sigma @ A.T + R

    K = Sigma_pred @ C.T @ np.linalg.inv(C @ Sigma_pred @ C.T + Q)
    mu_new = mu_pred + K @ (z - C @ mu_pred)
    Sigma_new = (np.eye(len(K)) - K @ C) @ Sigma_pred

    return (mu_new, Sigma_new)

mse_metrics = calc_mse(noisy_kalman_filter)
plot_mse(*mse_metrics)
