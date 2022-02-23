# OFC4HCI
OFC4HCI – Code for the paper "Optimal Feedback Control for Modeling Human-Computer Interaction" submitted to ToCHI

<large>**This python toolbox includes all OFC methods described in the paper "Optimal Feedback Control for Modeling Human-Computer Interaction" (2OL-Eq, MinJerk, LQR, LQG, and E-LQG).**</large>

**Usage:**
There are mainly three use cases this library is designed for:
1. **Compare** different OFC methods regarding their suitability to replicate observed pointing movements
    - Use `run_model_comparison.py`
    - Examples:
        - Compare 2OL-Eq and LQR with respect to rightward movements of User 2 in the 765x3-task, using optimal parameters:
        
          `python run_model_comparison.py 2OL-Eq LQR --user 2 --distance 765 --width 3 --direction right`
        - Compare LQR, LQG, and E-LQG with respect to leftward movements of User 12 in the 1275x85-task, using a signal-dependent noise level of 0.5 for both LQR and LQG, and default (non-optimal) parameters otherwise:
        
          `python run_model_comparison.py LQR LQG E-LQG --user 12 --distance 1275 --width 85 --direction left --lqg_params "sigma_u=0.5" --elqg_params "sigma_u=0.5" --no-opt`
        - Compare all available models using optimal parameters with respect to the default movement (rightward movements of User 3 in the 765x51-task, as shown in the paper):
        
          `python run_model_comparison.py 2OL-Eq MinJerk LQR LQG E-LQG`
          
2. **Optimize** parameters of an OFC method such that the resulting trajectories (or trajectory distributions) best match observed pointing movements with respect to a given loss function
    - Use `optimize_params.py`
    - Examples:
        - Optimize 2OL-Eq parameters "k" and "d" with respect to (positional) SSE, using the default user movement (rightward movements of User 3 in the 765x51-task, as shown in the paper) as reference:
        
          `python optimize_params.py 2OL-Eq k d --loss=SSE`
        - Optimize E-LQG parameters "r", "sigma_u", "sigma_Hdot", and "sigma_frc" with respect to Mean (2-)Wasserstein Distance (using positions and velocities by default), using rightward movements of User 2 in the 765x3-task as reference (other parameters are set to default values):
        
          `python optimize_params.py E-LQG r sigma_u sigma_Hdot sigma_frc --loss=MWD --user 2 --distance 765 --width 3 --direction right`
        - Optimize LQR parameter "r" with respect to (positional) SSE, using leftward movements of User 12 in the 1275x85-task, with "velocitycosts_weight" set to 0.005 and "forcecosts_weight" set to 0.0005 (other parameters are set to default values):
        
          `python optimize_params.py LQR r --loss=SSE --user 12 --distance 1275 --width 85 --direction left --params_fixed "velocitycosts_weight=0.005 forcecosts_weight=0.0005"`
          
3. **Use** control methods to simulate (optimal) user behavior in more general pointing tasks (applicability to other HCI tasks is discussed in the paper and requires some code modifications)
    - For each of the available models, the corresponding python script in "examples/" shows how to compute and visualize the model trajectory (or trajectory distribution) for a given initial position, target, and time horizon.
    - Examples:
        - `python examples/secondorderlag_eq_example.py`
        - `python examples/minjerk_example.py`
        - `python examples/lqr_example.py`
        - `python examples/lqg_example.py`
        - `python examples/elqg_example.py`

**Available control methods/models:**
- "2OL-Eq"
- "MinJerk" (adapted from [HoffArbib93](https://pubmed.ncbi.nlm.nih.gov/12581988/) and [Flash85](https://www.jneurosci.org/content/5/7/1688))
- "LQR"
- "LQG" (adapted from [Todorov05](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1550971/))
- "E-LQG" (adapted from [Todorov05](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1550971/), observation model adapted from [Todorov98_thesis](https://homes.cs.washington.edu/~todorov/papers/TodorovThesis.pdf))

**Model parameters:**
- "k": stiffness [2OL-Eq]
- "d": damping [2OL-Eq]
- "passage_times": array of indices that correspond to target passing times in via-point tasks; at these time steps, distance, velocity, and force costs are applied; here, this should be [0, N_MJ] for MinJerk, with N_MJ end time step of actual MinJerk trajectory, and [0, N] with N final time step index for LQG/E-LQG [MinJerk (see "N_MJ" in paper), LQG/E-LQG (not used in paper)]
- "r": negative log (!) of effort cost weight [LQR/LQG/E-LQG (corresponds to "-log(omega_r)" with "omega_r" from paper)]
- "velocitycosts_weight": velocity cost weight [LQR/LQG/E-LQG ("omega_v" in paper; note different usage between LQR and LQG/E-LQG!)]
- "forcecosts_weight": force cost weight [LQR/LQG/E-LQG ("omega_f" in paper; note different usage between LQR and LQG/E-LQG!)]
- "mass": mass of object to which forces are applied [LQR/LQG/E-LQG (not used in paper)]
- "t_const_1": time activation constant [LQR/LQG/E-LQG ("tau_1" in paper)]
- "t_const_2": time excitation constant [LQR/LQG/E-LQG ("tau_2" in paper)]
- "sigma_u": signal-dependent (multiplicative) control noise level [LQG/E-LQG]
- "sigma_c": constant (i.e., signal-independent) control noise level [LQG/E-LQG (not used in paper)]
- "sigma_s": observation noise scaling parameter [LQG]
- "sigma_H": proprioceptive position noise level (only used if "include_proprioceptive_endeffector_signals==True") [E-LQG (not used in paper)]
- "sigma_Hdot": velocity perception noise level [E-LQG ("sigma_v" in paper)]
- "sigma_frc": force perception noise level [E-LQG ("sigma_f" in paper)]
- "sigma_e": gaze noise level [E-LQG]
- "gamma": position perception noise weight [E-LQG]
- "saccade_times": array of indices that correspond to saccade times; here, this should be [n_s,N] (avoid whitespaces between list entries in command line argument!), with n_s saccade time between initial position and target and N final time step index [E-LQG (see "n_s" in paper)]
- "Delta": observation time lag in time steps (WARNING: experimental!) [LQG/E-LQG (not used in paper)]

**Loss functions:**
- "SSE" (positional Sum Squared Error)
- "Maximum Error" (positional Maximum Error)
- "MAE" (positional Mean Absolute Error; not included in model comparisons by default)
- "MKL" (Mean KL Divergence on position and velocity data; only for LQG and E-LQG!)
- "MWD" (Mean (2-)Wasserstein Distance on position and velocity data; only for LQG and E-LQG!)

**Requirements:**
- python>=3.8
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)

**Installing:**
- Clone this git and run `pip install -e .` from main directory.
- **Info:** Both `run_model_comparison.py` and `optimize_params.py` require the Processed Data of the [*Pointing Dynamics Dataset*](http://joergmueller.info/controlpointing/) from Müller et al. During first execution, you will be asked whether to download and unzip it automatically. Alternatively, you can specify its path manually.
