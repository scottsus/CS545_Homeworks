import numpy as np
import matplotlib.pyplot as plt
from plotters import plot_prediction_part_3

T = 20

"""
True values
"""

x0_true = 2
alpha_true = 0.1

"""
Initial estimates
"""

x0 = 1
alpha0 = 2
mu0 = np.array([[x0, alpha0]])
Sigma0 = np.diag([2, 2])

"""
Linearization
"""

def h(x) -> int:
    return np.sqrt(x**2 + 1)

"""
Noise terms
"""

noise_mean = 0
R = 0.5
Q = 1

def get_gaussian_noise(variance) -> int:
    return np.random.normal(noise_mean, np.sqrt(variance))

def extended_kalman_filter(mu, Sigma, z):
    x, alpha = mu[0][0], mu[1][0]
    A = np.array([
        [alpha, 0],
        [0, 1],
    ])
    H = np.array([x / np.sqrt(x**2 + 1), 0]).reshape(1, 2)

    mu_pred = A @ mu
    Sigma_pred = A @ Sigma @ A.T + np.array([
        [R, 0],
        [0, 0]
    ])

    K = Sigma_pred @ H.T @ np.linalg.inv(H @ Sigma_pred @ H.T + Q)
    mu_new = mu_pred + K.flatten() @ (z - h(mu_pred))
    Sigma_new = (np.eye(len(K)) - K @ H) @ Sigma_pred

    return mu_new, Sigma_new

def calc_prediction() -> tuple:
    mu = mu0.T
    Sigma = Sigma0

    timesteps = np.arange(0, T)
    ground_truths = [[x0_true, alpha_true]]
    predict_means = [mu]
    predict_cov = [Sigma]

    for t in range(1, T):
        x_true = alpha_true * ground_truths[t - 1][0] + get_gaussian_noise(R)
        ground_truths.append([x_true, alpha_true])

        x = mu[0]
        z = h(x) + get_gaussian_noise(Q)
        mu, Sigma = extended_kalman_filter(mu, Sigma, z)

        predict_means.append(mu)
        predict_cov.append(Sigma)
    
    ground_truths = np.stack(ground_truths)
    predict_means = np.stack(predict_means).reshape(-1, 2)
    predict_cov = np.stack(predict_cov)
    
    return timesteps, ground_truths, predict_means, predict_cov

prediction_metrics = calc_prediction()
plot_prediction_part_3(*prediction_metrics)
