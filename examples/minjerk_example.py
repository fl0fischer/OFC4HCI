import numpy as np
import matplotlib.pyplot as plt

import ofc4hci

# 1. Setup problem
N = 500
dt = 0.002
dim = 1
num_targets = 1
x0_pos = np.zeros((dim,))  #initial position
x0_vel = np.zeros((dim,))  #initial velocity
x0_acc = np.zeros((dim,))  #initial acceleration
x0 = np.concatenate((x0_pos, x0_vel, x0_acc))
x0_target = 0.2 * np.ones((dim*num_targets,))  #desired target position(s)
target_width = 0.01  #target radius (only for visualization)

minjerk_param_dict = {'passage_times': [0, 400]}

# 2. Solve problem
print(f"Compute MinJerk trajectory for N={N}, dt={dt}, x0={x0}, and T={x0_target}.\n"
      f"Parameter values (other than default):\n{minjerk_param_dict}")
x, u = ofc4hci.minjerk(N, dt, x0, dim, x0_target, **minjerk_param_dict)

# 3. Plot results
assert dim == 1, "Warning: Adjust plots for dim > 1!"

plt.figure(0)
plt.title(f"MinJerk Position Time Series")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.fill_between(np.arange(N+1)*dt, x0_target - target_width, x0_target + target_width, color='orange', alpha=.1, label="Target")
plt.plot(np.arange(N+1)*dt, x[:, 0], label="Position")
ymin, ymax = plt.gca().get_ylim()
plt.gca().set_ylim(ymin, ymax)
plt.vlines(minjerk_param_dict["passage_times"][1]*dt, ymin, ymax, color='black', linestyles="dashed")
plt.legend()

plt.figure(1)
plt.title(f"MinJerk Velocity Time Series")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.plot(np.arange(N+1)*dt, x[:, 1], label="Velocity")
ymin, ymax = plt.gca().get_ylim()
plt.gca().set_ylim(ymin, ymax)
plt.vlines(minjerk_param_dict["passage_times"][1]*dt, ymin, ymax, color='black', linestyles="dashed")
plt.legend()

# plt.figure(2)
# plt.title(f"MinJerk Control Time Series")
# plt.xlabel("Time (s)")
# plt.ylabel("Control")
# plt.plot(np.arange(N)*dt, u, label="Control")
# plt.legend()

plt.show()