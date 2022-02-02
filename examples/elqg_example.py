import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import ofc4hci

# 1. Setup problem
N = 500
dt = 0.002
dim = 1
num_targets = 2  #should be >= 2 for E-LQG
x0_pos = np.zeros((dim,))  #initial position
x0_vel = np.zeros((dim,))  #initial velocity
x0_musclestate = np.ones((2*dim,))  #initial muscle activation and excitation
x0_init_target = np.zeros((dim,))  #initial (target) position
x0_target = 0.2 * np.ones((dim*(num_targets-1),))  #desired target position(s)
target_width = 0.01  #target radius (only for visualization)
x0_mean = np.concatenate((x0_pos, x0_vel, x0_musclestate, x0_init_target, x0_target))
u0 = np.array([0])  #unused here
n = dim * (4 + num_targets)
Sigma0 = np.zeros((n, n))

lqg_param_dict = {'r': 30,  #negative log of omega_r from paper
 'velocitycosts_weight': 0.003,
 'forcecosts_weight': 0.001,
 'sigma_u': 2.5,
 'sigma_Hdot': 0.3,
 'sigma_frc': 3,
 'sigma_e': 4,
 'gamma': 0.02,
 'saccade_times': [250, N],
 'mass': 1,
 't_const_1': 0.04,
 't_const_2': 0.04}

confidence_interval_percentage = 0.95  #only for visualization

# 2. Solve problem
print(f"Compute E-LQG trajectory for N={N}, dt={dt}, x0_mean={x0_mean}, Sigma0={Sigma0}, and T={x0_target}.\n"
      f"Parameter values (other than default):\n{lqg_param_dict}")
J, x_expectation, Sigma_x, u_expectation, Ical, x, u, y, L, Kk = ofc4hci.lqg(N, dt, x0_mean, u0, Sigma0, dim, num_targets,
                                                        system_dynamics="E-LQG", modify_init_target_estimate=True,
                                                        **lqg_param_dict)
print(f"Optimal costs: {J}.")

# 3. Plot results
assert dim == 1, "Warning: Adjust plots for dim > 1!"

plt.figure(0)
plt.title(f"E-LQG Position Time Series")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.fill_between(np.arange(N+1)*dt, x0_target[0] - target_width, x0_target[0] + target_width, color='orange', alpha=.1, label="Target")
plt.plot(np.arange(N+1)*dt, x[:, 0], label="True Position")
plt.plot(np.arange(N+1)*dt, Ical[:, 0], label="Internal Position Estimate")
plt.plot(np.arange(N+1)*dt, x_expectation[:, 0], label="Expected Internal Position Estimate")
pos_std = np.array([np.sqrt(Sigma_x[i][0, 0]) for i in range(N+1)])
plt.fill_between(np.arange(N+1)*dt, x_expectation[:, 0] - stats.norm.ppf(
    (confidence_interval_percentage + 1) / 2) * pos_std, x_expectation[:, 0] + stats.norm.ppf(
    (confidence_interval_percentage + 1) / 2) * pos_std, color='darkseagreen', alpha=.1, label="95%-CI")
plt.legend()

plt.figure(1)
plt.title(f"E-LQG Velocity Time Series")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.plot(np.arange(N+1)*dt, x[:, 1], label="True Velocity")
plt.plot(np.arange(N+1)*dt, Ical[:, 1], label="Internal Velocity Estimate")
plt.plot(np.arange(N+1)*dt, x_expectation[:, 1], label="Expected Internal Velocity Estimate")
vel_std = np.array([np.sqrt(Sigma_x[i][1, 1]) for i in range(N+1)])
plt.fill_between(np.arange(N+1)*dt, x_expectation[:, 1] - stats.norm.ppf(
    (confidence_interval_percentage + 1) / 2) * vel_std, x_expectation[:, 1] + stats.norm.ppf(
    (confidence_interval_percentage + 1) / 2) * vel_std, color='darkseagreen', alpha=.1, label="95%-CI")
plt.legend()

plt.figure(2)
plt.title(f"E-LQG Control Time Series")
plt.xlabel("Time (s)")
plt.ylabel("Control")
plt.plot(np.arange(N)*dt, u, label="Applied Control")
plt.plot(np.arange(N)*dt, u_expectation, color="green", label="Expected Applied Control")
plt.legend()

plt.show()