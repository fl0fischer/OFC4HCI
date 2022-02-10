import os.path
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import expm, eigh, sqrtm
from scipy.optimize import differential_evolution

from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen

from pd_dataset import datasetpreparation


def pointingdynamics_data(user, distance, width, direction,
                          manual_reactiontime_cutoff=True,
                          shift_target_to_origin=False, analysis_dim=2,
                          dir_path="PointingDynamicsDataset", DATAtrials=None, trial_info_path="PD-Dataset.pickle"):
    """
    Computes reference trajectory (distribution) from Pointing Dynamics Dataset.
    :param user: user ID [1-12]
    :param distance: distance to target in px [765 or 1275]
    :param width: corresponding target width in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}]
    :param direction: movement direction ["left" or "right"]
    :param manual_reactiontime_cutoff: whether reaction times should be removed from reference user trajectory [bool]
    :param shift_target_to_origin: whether to shift coordinate system such that target center is in origin [bool]
    :param analysis_dim: how many dimensions of user data trajectories should be used for mean computation etc. (e.g., 2 means position and velocity only) [1-3]
    :param dir_path: local path to Mueller's PointingDynamicsDataset (only used if "dir_path.txt" does not exist) [str]
    :param DATAtrials: trial info object (if not provided, this is loaded from trial_info_path) [pandas.DataFrame]
    :param trial_info_path: path to trial info object (if file does not exist, DATAtrials object is re-computed and stored to file) [str]
    :return:
        - x_loc_data: average user trajectory [(N+1)*analysis_dim-array]
        - x_scale_data: covariance sequence of user trajectory [(N+1)*analysis_dim*analysis_dim-array]
        - Nmax: number of time steps (excluding final step; computed as longest duration of considered trials) [int]
        - dt: time step duration in seconds [int/float]
        - dim: dimension of the task (1D, 2D, or 3D) [int]
        - initialvalues_alltrials: list of tuples containing (actual) initial position, velocity, and acceleration for each considered user trial [list of tuples]
        - T_alltrials: list of tuples containing (nominal) initial position (=target position of preceding trial) and current target position for each considered user trial [list of tuples]
        - widthm_alltrials: list of tuples containing width (i.e., radius) of preceding and current target spheres for each considered user trial [list of tuples]
    """

    # 1. Read dataset and meta information
    # Provide local path to Mueller's PointingDynamicsDataset
    # (http://joergmueller.info/controlpointing/PointingDynamicsDataset.zip)
    if os.path.isfile("dir_path.txt"):  # if dir_path has been stored from a previous run, use this
        with open("dir_path.txt", "r") as file:
            dir_path = file.read()
        dir_path = os.path.abspath(dir_path)
    else:
        dir_path = os.path.abspath(dir_path)
        if not os.path.exists(dir_path):
            download_PointingDynamicsDataset = input("Could not find reference to PointingDynamicsDataset. Do you want to download it (~1.4GB)? (y/N) ")
            if download_PointingDynamicsDataset.lower().startswith("y"):
                print("Will download and unzip it to 'PointingDynamicsDataset/'.")
                print("Downloading archive... ", end='', flush=True)
                resp = urlopen("http://joergmueller.info/controlpointing/PointingDynamicsDataset.zip")
                zipfile = ZipFile(BytesIO(resp.read()))
                print("unzip archive... ", end='', flush=True)
                for file in zipfile.namelist():
                    zipfile.extract(file)
                print("done.")
                dir_path = os.path.abspath("PointingDynamicsDataset")
                assert os.path.exists(dir_path), "Internal Error during unpacking of PointingDynamicsDataset."
            else:
                dir_path = input("Insert the path to the PointingDynamicsDataset directory: ")
                dir_path = os.path.abspath(dir_path)
                assert os.path.exists(dir_path), "ERROR: Invalid input path."

            with open("dir_path.txt", "w") as file:  # remember dir_path for later runs
                file.write(dir_path)

    dim = 1  # dimension of the task (1D, 2D, or 3D)

    if DATAtrials is None:
        if trial_info_path is not None and os.path.exists(trial_info_path):
            # -> OPTION 1: load trial information from file
            with open(trial_info_path, 'rb') as handle:
                DATAtrials = pickle.load(handle)
        else:
            # -> OPTION 2: compute and store trial information
            DATAtrials = datasetpreparation(dir_path=dir_path)
            with open('PD-Dataset.pickle', 'wb') as handle:
                pickle.dump(DATAtrials, handle, protocol=pickle.HIGHEST_PROTOCOL)

    filenameread = os.path.join(dir_path,f"P{user}",f"Data0,-1,{distance},{width}.csv")
    
    assert os.path.isfile(filenameread), "Pointing data could not be found at given path. Make sure that 'dir_path.txt' in the project's main folder contains the correct path or delete the file."
    
    DATA = pd.read_csv(filenameread)
    dt = np.mean(np.diff(DATA.time))  # DATA time step duration

    # 2. Split and pre-process dataset
    available_trials = DATAtrials.loc[(DATAtrials['User'] == user) & (DATAtrials['Distance'] == distance) & (
            DATAtrials['Width'] == width) & (DATAtrials['Direction'] == direction), 'N0':'N1']
    N0_alltrials = available_trials['N0'].values[0]
    ndelta_alltrials = available_trials['ndelta'].values[0]
    N1_alltrials = available_trials['N1'].values[0]
    T_alltrials = [[[target_pos for target_pos in DATA.target.unique() if target_pos != DATA.target[N0 + 1]][0],
                    DATA.target[N0 + 1]] for N0 in N0_alltrials]
    widthm_alltrials = [[DATA.widthm[N0 + 1], DATA.widthm[N0 + 1]] for N0 in N0_alltrials]
    # Remove reaction times
    if manual_reactiontime_cutoff:
        initialvalues_alltrials = [(DATA.y[ndelta], DATA.sgv[ndelta], DATA.sga[ndelta]) for ndelta in ndelta_alltrials]
        ydata_alltrials = [
            np.array([DATA.y[ndelta:N1 + 1], DATA.sgv[ndelta:N1 + 1], DATA.sga[ndelta:N1 + 1]]).transpose() for
            ndelta, N1 in zip(ndelta_alltrials, N1_alltrials)]
    else:
        initialvalues_alltrials = [(DATA.y[N0], DATA.sgv[N0], DATA.sga[N0]) for N0 in N0_alltrials]
        ydata_alltrials = [np.array([DATA.y[N0:N1 + 1], DATA.sgv[N0:N1 + 1], DATA.sga[N0:N1 + 1]]).transpose() for
                           N0, N1 in zip(N0_alltrials, N1_alltrials)]
    # Translate coordinates such that target is in origin
    if shift_target_to_origin:
        initialvalues_alltrials = [
            (initialvalues_trial[0] - T_trial[-1], initialvalues_trial[1], initialvalues_trial[2]) for
            initialvalues_trial, T_trial in zip(initialvalues_alltrials, T_alltrials)]
        ydata_alltrials = [ydata_trial - [T_trial[-1], 0, 0] for ydata_trial, T_trial in
                           zip(ydata_alltrials, T_alltrials)]
        T_alltrials = [list(np.array(T_trial) - T_trial[-1]) for T_trial in T_alltrials]
    # Remove trajectories with any outlier position value
    ydata_alltrials = pd.Series(ydata_alltrials)[np.bitwise_and.reduce(np.squeeze([[np.abs(stats.zscore(
        [ydata_trial[i, 0] if i < len(ydata_trial) else ydata_trial[-1, 0] for ydata_trial in ydata_alltrials])) < 3]
                                                                                   for i in range(
            max([len(ydata_trial) for ydata_trial in ydata_alltrials]))]))].tolist()
    # Remove trajectories with outlier duration
    ydata_alltrials = pd.Series(ydata_alltrials)[
        np.abs(stats.zscore([ydata_trial.shape[0] for ydata_trial in ydata_alltrials])) < 3].tolist()

    # 3. Compute mean trajectories
    # (Extend all trajectories to longest duration of considered trials (constantly using last value and zero velocity))
    Nmax = max([ydata_trial.shape[0] for ydata_trial in ydata_alltrials]) - 1
    x_loc_data = [None] * (Nmax + 1)  # np.zeros(shape=(Nmax, ))
    x_scale_data = [None] * (Nmax + 1)  # np.zeros(shape=(Nmax, ))
    for i in range(Nmax + 1):
        x_loc_data[i] = np.mean(np.squeeze(
            [ydata_trial[i, :analysis_dim] if ydata_trial.shape[0] > i else
             (ydata_trial[-1, :] * np.repeat(np.array([1, 0, 0]), dim))[:analysis_dim] for
             ydata_trial in ydata_alltrials]).reshape(-1, analysis_dim), axis=0)
        x_scale_data[i] = np.cov(np.squeeze(
            [ydata_trial[i, :analysis_dim] if ydata_trial.shape[0] > i else
             (ydata_trial[-1, :] * np.repeat(np.array([1, 0, 0]), dim))[:analysis_dim] for
             ydata_trial in ydata_alltrials]), rowvar=0).reshape(-1, analysis_dim)
    x_loc_data = np.squeeze(x_loc_data)
    x_scale_data = np.squeeze(x_scale_data)

    return x_loc_data, x_scale_data, Nmax, dt, dim, initialvalues_alltrials, T_alltrials, widthm_alltrials


def _control_pointingdynamics(control_model, *args, **kwargs):
    """
    Wrapper that passes args and kwargs to model-specific pointingdynamics function.
    :param control_model: control model to use [str]
    :param args: arbitrary number of arguments
    :param kwargs: arbitrary number of keyword arguments
    :return: output of respective model function
    """
    if control_model == "2OL-Eq":
        return secondorderlag_eq_pointingdynamics(*args, **kwargs)
    elif control_model == "MinJerk":
        return minjerk_pointingdynamics(*args, **kwargs)
    elif control_model == "LQR":
        return lqr_pointingdynamics(*args, **kwargs)
    elif control_model == "LQG":
        return lqg_pointingdynamics(*args, **kwargs, system_dynamics="LQG")
    elif control_model == "E-LQG":
        return lqg_pointingdynamics(*args, **kwargs, system_dynamics="E-LQG")
    else:
        raise NotImplementedError


def _get_custom_param_dict(control_model, user, distance, width, direction, param_dict_custom=None, use_opt_params=False):
    """
    Sets up parameter dict consisting of custom and optimal parameter values (if "use_opt_params==True").
    :param control_model: control model to use (only used if "use_opt_params==True") [str]
    :param user: user ID (only used if "use_opt_params==True") [1-12]
    :param distance: distance to target in px (only used if "use_opt_params==True") [765 or 1275]
    :param width: corresponding target width in px (only used if "use_opt_params==True")
     [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}]
    :param direction: movement direction (only used if "use_opt_params==True") ["left" or "right"]
    :param param_dict_custom: dictionary with name-value pairs of "control_models" parameters [dict]
    :param use_opt_params: whether optimal params should be used for non-specified parameters
     (otherwise, these parameters are omitted from returned dictionary, i.e., their default values will be used) [bool]
    :return: parameter dictionary [dict]
    """
    # Load values of non-specified parameters (either default or optimal parameters)
    if use_opt_params:
        with open(f"optparams/{control_model}/optparam_{user}_{distance}-{width}_{direction}.npy", 'rb') as f:
            opt_info_npy = np.load(f, allow_pickle=True)
        param_dict = opt_info_npy[1]
    else:
        param_dict = {}
    # Update dict with specified parameter values
    if param_dict_custom is not None:
        param_dict.update(param_dict_custom)

    return param_dict


def secondorderlag_eq_pointingdynamics(user, distance, width, direction, secondorderlag_eq_param_dict,
                          manual_reactiontime_cutoff=True,
                          shift_target_to_origin=False, analysis_dim=1,
                          dir_path="PointingDynamicsDataset", DATAtrials=None, trial_info_path="PD-Dataset.pickle"):
    """
    Computes 2OL-Eq trajectory for a given parameter set, as well as several metrics comparing this trajectory to
     reference user trajectory from Pointing Dynamics Dataset.
    :param user: user ID [1-12]
    :param distance: distance to target in px [765 or 1275]
    :param width: corresponding target width in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}]
    :param direction: movement direction ["left" or "right"]
    :param secondorderlag_eq_param_dict: dictionary with name-value pairs of 2OL-Eq parameters; missing parameters are set to their default value [dict]
    :param manual_reactiontime_cutoff: whether reaction times should be removed from reference user trajectory [bool]
    :param shift_target_to_origin: whether to shift coordinate system such that target center is in origin [bool]
    :param analysis_dim: how many dimensions of user data trajectories should be used for mean computation etc. (e.g., 2 means position and velocity only) [1-2]
    :param dir_path: local path to Mueller's PointingDynamicsDataset (only used if "dir_path.txt" does not exist) [str]
    :param DATAtrials: trial info object (if not provided, this is loaded from trial_info_path) [pandas.DataFrame]
    :param trial_info_path: path to trial info object (if file does not exist, DATAtrials object is re-computed and stored to file) [str]
    :return:
        - x: resulting state sequence [(N+1)*n-array]
        - u: resulting control sequence [N*dim-array]
        - x_loc_data: average user trajectory [(N+1)*analysis_dim-array]
        - SSE: Sum Squared Error between 2OL-Eq and user trajectory (along first analysis_dim dimensions) [float]
        - MaximumError: Maximum Error between 2OL-Eq and user trajectory (along first analysis_dim dimensions) [float]
    """

    assert analysis_dim in [1, 2], "Invalid analyis_dim."

    # 1. Compute task information and reference data trajectories/distributions
    x_loc_data, x_scale_data, Nmax, dt, dim, \
    initialvalues_alltrials, T_alltrials, widthm_alltrials = pointingdynamics_data(user, distance, width, direction,
                                                              manual_reactiontime_cutoff=manual_reactiontime_cutoff,
                                                              shift_target_to_origin=shift_target_to_origin,
                                                              analysis_dim=analysis_dim,
                                                              dir_path=dir_path, DATAtrials=DATAtrials, trial_info_path=trial_info_path)

    # 2. Compute initial state x0 (if not given)
    T_alltrials = [T[1:] for T in T_alltrials]  # remove initial position from target T (and thus from initial state x0)
    x0_alltrials = [np.array([initialuservalues_trial[0], initialuservalues_trial[1]])
                    for initialuservalues_trial in initialvalues_alltrials]
    x0 = np.mean(x0_alltrials, axis=0)
    assert len(np.unique(T_alltrials)) == 1, "ERROR: Target changes during movement..."
    T = T_alltrials[0]

    # 3. Run 2OL-Eq
    x, u = secondorderlag_eq(Nmax, dt, x0, dim, T, **secondorderlag_eq_param_dict)

    # 4. Compute metrics that compare simulation and user data
    x_loc_sim = x[:, :analysis_dim]
    SSE = compute_SSE(x_loc_sim, x_loc_data)
    MaximumError = compute_MaximumError(x_loc_sim, x_loc_data)

    return x, u, x_loc_data, SSE, MaximumError


def minjerk_pointingdynamics(user, distance, width, direction, minjerk_param_dict,
                          manual_reactiontime_cutoff=True,
                          shift_target_to_origin=False, analysis_dim=1,
                          dir_path="PointingDynamicsDataset", DATAtrials=None, trial_info_path="PD-Dataset.pickle"):
    """
    Computes MinJerk trajectory for a given parameter set, as well as several metrics comparing this trajectory to
     reference user trajectory from Pointing Dynamics Dataset.
    :param user: user ID [1-12]
    :param distance: distance to target in px [765 or 1275]
    :param width: corresponding target width in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}]
    :param direction: movement direction ["left" or "right"]
    :param minjerk_param_dict: dictionary with name-value pairs of MinJerk parameters; missing parameters are set to their default value [dict]
    :param manual_reactiontime_cutoff: whether reaction times should be removed from reference user trajectory [bool]
    :param shift_target_to_origin: whether to shift coordinate system such that target center is in origin [bool]
    :param analysis_dim: how many dimensions of user data trajectories should be used for mean computation etc. (e.g., 2 means position and velocity only) [1-3]
    :param dir_path: local path to Mueller's PointingDynamicsDataset (only used if "dir_path.txt" does not exist) [str]
    :param DATAtrials: trial info object (if not provided, this is loaded from trial_info_path) [pandas.DataFrame]
    :param trial_info_path: path to trial info object (if file does not exist, DATAtrials object is re-computed and stored to file) [str]
    :return:
        - x: resulting state sequence [(N+1)*n-array]
        - u: resulting control sequence [N*dim-array]
        - x_loc_data: average user trajectory [(N+1)*analysis_dim-array]
        - SSE: Sum Squared Error between MinJerk and user trajectory (along first analysis_dim dimensions) [float]
        - MaximumError: Maximum Error between MinJerk and user trajectory (along first analysis_dim dimensions) [float]
    """

    # 1. Compute task information and reference data trajectories/distributions
    x_loc_data, x_scale_data, Nmax, dt, dim, \
    initialvalues_alltrials, T_alltrials, widthm_alltrials = pointingdynamics_data(user, distance, width, direction,
                                                              manual_reactiontime_cutoff=manual_reactiontime_cutoff,
                                                              shift_target_to_origin=shift_target_to_origin,
                                                              analysis_dim=analysis_dim,
                                                              dir_path=dir_path, DATAtrials=DATAtrials, trial_info_path=trial_info_path)

    # 2. Compute initial state x0 (if not given)
    T_alltrials = [T[1:] for T in T_alltrials]  # remove initial position from target T (and thus from initial state x0)
    x0_alltrials = [np.array(list(initialuservalues_trial)) for initialuservalues_trial in initialvalues_alltrials]
    x0 = np.mean(x0_alltrials, axis=0)
    assert len(np.unique(T_alltrials)) == 1, "ERROR: Target changes during movement..."
    T = T_alltrials[0]

    # 3. Run MinJerk
    x, u = minjerk(Nmax, dt, x0, dim, T, **minjerk_param_dict)

    # 4. Compute metrics that compare simulation and user data
    x_loc_sim = x[:, :analysis_dim]
    SSE = compute_SSE(x_loc_sim, x_loc_data)
    MaximumError = compute_MaximumError(x_loc_sim, x_loc_data)

    return x, u, x_loc_data, SSE, MaximumError


def lqr_pointingdynamics(user, distance, width, direction, lqr_param_dict,
                          manual_reactiontime_cutoff=True,
                          shift_target_to_origin=False, analysis_dim=1,
                          dir_path="PointingDynamicsDataset", DATAtrials=None, trial_info_path="PD-Dataset.pickle"):
    """
    Computes LQR trajectory for a given parameter set, as well as several metrics comparing this trajectory to
     reference user trajectory from Pointing Dynamics Dataset.
    :param user: user ID [1-12]
    :param distance: distance to target in px [765 or 1275]
    :param width: corresponding target width in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}]
    :param direction: movement direction ["left" or "right"]
    :param lqr_param_dict: dictionary with name-value pairs of LQR parameters; missing parameters are set to their default value [dict]
    :param manual_reactiontime_cutoff: whether reaction times should be removed from reference user trajectory [bool]
    :param shift_target_to_origin: whether to shift coordinate system such that target center is in origin [bool]
    :param analysis_dim: how many dimensions of user data trajectories should be used for mean computation etc. (e.g., 2 means position and velocity only) [1-3]
    :param dir_path: local path to Mueller's PointingDynamicsDataset (only used if "dir_path.txt" does not exist) [str]
    :param DATAtrials: trial info object (if not provided, this is loaded from trial_info_path) [pandas.DataFrame]
    :param trial_info_path: path to trial info object (if file does not exist, DATAtrials object is re-computed and stored to file) [str]
    :return:
        - J: optimal costs [float]
        - x: optimal state sequence [(N+1)*n-array]
        - u: optimal control sequence [N*dim-array]
        - x_loc_data: average user trajectory [(N+1)*analysis_dim-array]
        - SSE: Sum Squared Error between LQR and user trajectory (along first analysis_dim dimensions) [float]
        - MaximumError: Maximum Error between LQR and user trajectory (along first analysis_dim dimensions) [float]
    """

    # 1. Compute task information and reference data trajectories/distributions
    x_loc_data, x_scale_data, Nmax, dt, dim, \
    initialvalues_alltrials, T_alltrials, widthm_alltrials = pointingdynamics_data(user, distance, width, direction,
                                                              manual_reactiontime_cutoff=manual_reactiontime_cutoff,
                                                              shift_target_to_origin=shift_target_to_origin,
                                                              analysis_dim=analysis_dim,
                                                              dir_path=dir_path, DATAtrials=DATAtrials, trial_info_path=trial_info_path)
    num_targets = 1  # number of via-point targets (default pointing task: 1)

    # 2. Compute initial state x0 (if not given)
    T_alltrials = [T[1:] for T in T_alltrials]  # remove initial position from target T (and thus from initial state x0)
    x0_alltrials = [np.array([initialuservalues_trial[0], initialuservalues_trial[1]] + [0] * (2 * dim) + T_trial)
                    for (T_trial, initialuservalues_trial) in zip(T_alltrials, initialvalues_alltrials)]
    x0 = np.mean(x0_alltrials, axis=0)

    # 3. Run LQR
    J, x, u = lqr(Nmax, dt, x0, dim, **lqr_param_dict, num_targets=num_targets)

    # 4. Compute metrics that compare simulation and user data
    x_loc_sim = x[:, :analysis_dim]
    SSE = compute_SSE(x_loc_sim, x_loc_data)
    MaximumError = compute_MaximumError(x_loc_sim, x_loc_data)

    return J, x, u, x_loc_data, SSE, MaximumError


def lqg_pointingdynamics(user, distance, width, direction, lqg_param_dict,
                          system_dynamics="LQG",
                          manual_reactiontime_cutoff=True,
                          shift_target_to_origin=False, analysis_dim=2, analysis_dim_deterministic=1,
                          algorithm_iterations=20, J_eps=1e-3,
                          include_proprioceptive_target_signals=False,
                          include_proprioceptive_endeffector_signals=False,
                          dir_path="PointingDynamicsDataset", DATAtrials=None, trial_info_path="PD-Dataset.pickle"):
    """
    Computes LQG or E-LQG trajectory for a given parameter set, and compares it to reference data from Pointing Dynamics Dataset using various metrics.
    :param user: user ID [1-12]
    :param distance: distance to target in px [765 or 1275]
    :param width: corresponding target width in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}]
    :param direction: movement direction ["left" or "right"]
    :param lqg_param_dict: dictionary with name-value pairs of LQG/E-LQG parameters; missing parameters are set to their default value [dict]
    :param system_dynamics: which dynamics to use ["LQG" or "E-LQG"]
    :param manual_reactiontime_cutoff: whether reaction times should be removed from reference user trajectory [bool]
    :param shift_target_to_origin: whether to shift coordinate system such that target center is in origin [bool]
    :param analysis_dim: how many dimensions of user data trajectories should be used for mean computation, stochastic measures, etc. (e.g., 2 means position and velocity only) [1-3]
    :param analysis_dim_deterministic: how many dimensions of user data trajectories should be used for deterministic measures
    (e.g., for comparisons with 2OL-Eq, MinJerk, and LQR; e.g., 2 means position and velocity only) [1-3]
    :param algorithm_iterations: (maximum) number of iterations, where the optimal control and
    the optimal observation problem is solved alternately (if "J_eps" is set, early termination is possible) [int]
    :param J_eps: if relative improvement of cost function falls below "J_eps" and "min_algorithm_iterations" is reached,
    iterative solving algorithm terminates [float]
    :param include_proprioceptive_target_signals: whether target position(s) can be observed in absolute coordinates
    (only usable for LQG; default: False) [bool]
    :param include_proprioceptive_endeffector_signals: whether end-effector can be observed position in absolute coordinates
    (only usable for E-LQG; default: False) [bool]
    :param dir_path: local path to Mueller's PointingDynamicsDataset (only used if "dir_path.txt" does not exist) [str]
    :param DATAtrials: trial info object (if not provided, this is loaded from trial_info_path) [pandas.DataFrame]
    :param trial_info_path: path to trial info object (if file does not exist, DATAtrials object is re-computed and stored to file) [str]
    :return:
        - Ical_expectation: mean of expected optimal state (or rather internal state estimate) sequence [(N+1)*m-array]
        - Sigma_x: variance of expected optimal (true) state sequence [list of (N+1) m*m-arrays]
        - x_loc_data: average user trajectory [(N+1)*analysis_dim-array]
        - x_scale_data: covariance sequence of user trajectory [(N+1)*analysis_dim*analysis_dim-array]
        - SSE: Sum Squared Error between LQG/E-LQG and user trajectory (along first analysis_dim dimensions) [float]
        - MaximumError: Maximum Error between LQR and user trajectory (along first analysis_dim dimensions) [float]
        - MKL: Mean KL Divergence between LQG/E-LQG and user trajectory (along first analysis_dim dimensions) [float]
        - MWD: Mean 2-Wasserstein Distance between LQG/E-LQG and user trajectory (along first analysis_dim dimensions) [float]
    """

    # 1. Compute task information and reference data trajectories/distributions
    x_loc_data, x_scale_data, Nmax, dt, dim, \
    initialvalues_alltrials, T_alltrials, widthm_alltrials = pointingdynamics_data(user, distance, width, direction,
                                                              manual_reactiontime_cutoff=manual_reactiontime_cutoff,
                                                              shift_target_to_origin=shift_target_to_origin,
                                                              analysis_dim=analysis_dim,
                                                              dir_path=dir_path, DATAtrials=DATAtrials, trial_info_path=trial_info_path)
    num_targets = 1 + (system_dynamics == "E-LQG")  # number of via-point targets (default pointing task: 1 (2 if system_dynamics == "E-LQG", as initial position needs to be included then))


    # 2. Compute initial state mean "x0_mean" and covariance "Sigma0" (if not given)
    T_alltrials = [T[(system_dynamics == "LQG"):] for T in T_alltrials]  # remove initial position from target T (and thus from initial state x0; only for LQG!)
    x0_alltrials = [np.array([initialuservalues_trial[0], initialuservalues_trial[1]] + [0] * (2 * dim) + T_trial)
                    for (T_trial, initialuservalues_trial) in zip(T_alltrials, initialvalues_alltrials)]
    x0_mean = np.mean(x0_alltrials, axis=0)
    u0 = np.zeros(shape=(dim, ))
    # Define initial covariance matrix (the variable Sigma0 corresponds to Sigma_1 in [Todorov1998])
    Sigma0 = np.cov(x0_alltrials, rowvar=False)
    Sigma0[np.abs(Sigma0) < 10e-10] = 0


    (_, x_expectation, Sigma_x, _, _, _, _, _, _, _) = lqg(Nmax, dt, x0_mean, u0, Sigma0, dim, num_targets,
                                                        system_dynamics=system_dynamics,
                                                        include_proprioceptive_target_signals=include_proprioceptive_target_signals,
                                                        include_proprioceptive_endeffector_signals=include_proprioceptive_endeffector_signals,
                                                        minimum_computations=True,
                                                        algorithm_iterations=algorithm_iterations,
                                                        J_eps=J_eps,
                                                        **lqg_param_dict)

    # 3. Compute metrics that compare simulation and user data
    x_loc_sim = x_expectation[:, :analysis_dim]
    x_scale_sim = np.squeeze([cov_matrix[:analysis_dim, :analysis_dim] for cov_matrix in Sigma_x])

    x_loc_sim_deterministic = x_expectation[:, :analysis_dim_deterministic]
    x_loc_data_deterministic = x_loc_data[:, :analysis_dim_deterministic]

    SSE = compute_SSE(x_loc_sim_deterministic, x_loc_data_deterministic)
    MaximumError = compute_MaximumError(x_loc_sim_deterministic, x_loc_data_deterministic)
    MKL = compute_MKL_normal(x_loc_data, x_scale_data, x_loc_sim, x_scale_sim)  #WARNING: order of arguments matter, since MKL is no real metric!
    MWD = compute_MWD_normal(x_loc_data, x_scale_data, x_loc_sim, x_scale_sim)

    return x_expectation, Sigma_x, x_loc_data, x_scale_data, SSE, MaximumError, MKL, MWD


def secondorderlag_eq(N, dt, x0, dim, T,
                      k=50, d=30):
    """
    Computes 2OL-Eq trajectory of given duration, using dynamics as described in the paper.
    :param N: number of time steps (excluding final step) [int]
    :param dt: time step duration in seconds [h in paper] [int/float]
    :param x0: initial state ((flattened) position and velocity) [list/array of length 2*dim]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param num_targets: number of targets (does not have an actual effect right now, but could be used for via-point tasks) [int]
    :param T: target position [list/array of length dim]
    :param k: stiffness parameter [float]
    :param d: damping parameter [float]
    :return:
        - x: resulting state sequence [(N+1)*n-array]
        - u: resulting control sequence [N*dim-array]
    """

    # 1. Compute required system matrices
    n = dim * 2  # dimension of state vector (incorporating position and velocity)
    ## VARIANT 1: use forward Euler approximation of 2OL
    # A = np.vstack((np.hstack((np.eye(dim), dt * np.eye(dim))),
    #                np.hstack(((-k * dt) * np.eye(dim), (1 - d * dt) * np.eye(dim)))))
    # B = np.vstack((np.zeros(shape=(dim, dim)), dt * np.eye(dim)))
    ## VARIANT 2: use exact transformation from continuous to discrete 2OL system matrices
    A_cont = np.vstack((np.hstack((np.zeros(shape=(dim, dim)), np.eye(dim))),
                        np.hstack((-k * np.eye(dim), -d * np.eye(dim)))))
    B_cont = np.vstack((np.zeros(shape=(dim, dim)), np.eye(dim)))
    A = expm(A_cont * dt)
    B = np.linalg.pinv(A_cont).dot(A - np.eye(n)).dot(B_cont)
    assert len(x0) == n, "Initial state x0 has wrong dimension!"

    # 2. (Online) Control Algorithm
    x = np.zeros((N + 1, n))
    u = np.zeros((N, dim))

    x[0] = x0
    for i in range(0, N):
        u[i] = k * np.array(T)
        x[i + 1] = A.dot(x[i]) + B.dot(u[i])

    return x, u


def minjerk(N, dt, x0, dim, T,
            final_vel=None, final_acc=None,
            passage_times=None):
    """
    Computes MinJerk trajectory of given duration (actually, Minjerk is only applied until passage_times[-1] and constantly extended afterwards).
    :param N: number of time steps (excluding final step) [int]
    :param dt: time step duration in seconds [h in paper] [int/float]
    :param x0: initial state ((flattened) position, velocity, acceleration) [list/array of length 3*dim]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param T: target position [list/array of length dim]
    :param final_vel: desired terminal velocity (if this is set, a different algorithm is used to compute MinJerk trajectory) [list/array of length dim]
    :param final_acc: desired terminal acceleration (if this is set, a different algorithm is used to compute MinJerk trajectory) [list/array of length dim]
    :param passage_times: array of indices that correspond to target passing times;
    here, this should be [0, N_MJ], with N_MJ end time step of actual MinJerk trajectory (can be continuous) [2-array]
    :return:
        - x: resulting state sequence [(N+1)*n-array]
        - u: resulting control sequence [N*dim-array]
    """

    # 1. Compute required system matrices
    if passage_times is None:
        passage_times = np.linspace(0, N, 2).astype(
            int)  # WARNING: here: equally distributed target passage times!
    assert len(passage_times) == 2
    n = dim * 3  # dimension of state vector (incorporating position, velocity, acceleration, and target position(s))

    N_MJ = np.ceil(passage_times[1]).astype(int)


    # 2. (Online) Control Algorithm
    x = np.zeros((N + 1, n))
    u = np.zeros((N, dim))

    if (final_vel is None) and (final_acc is None):
        ### VARIANT 1:  (this variant terminates with zero velocity and acceleration, independent of "final_vel" and "final_acc"!)
        A = [None] * (N_MJ)
        B = [None] * (N_MJ)

        x[0] = x0
        for i in range(0, N_MJ):
            u[i] = T

            # Compute time-dependent system matrices (with time relexation due to usage of passage_times[1] instead of int(passage_times[1])) [HoffArbib93]
            movement_time = (passage_times[1] - i) * dt
            A_continuous = np.vstack(
                (np.hstack((np.zeros(shape=(dim, dim)), np.eye(dim), np.zeros(shape=(dim, dim)))),
                 np.hstack((np.zeros(shape=(dim, 2 * dim)), np.eye(dim))),
                 np.hstack(((-60 / (movement_time ** 3)) * np.eye(dim), (-36 / (movement_time ** 2)) * np.eye(dim),
                            (-9 / movement_time) * np.eye(dim)))))
            B_continuous = np.vstack((np.zeros(shape=(2 * dim, dim)), (60 / (movement_time ** 3)) * np.eye(dim)))
            # Use explicit solution formula
            A[i] = expm(A_continuous * dt)
            B[i] = np.linalg.pinv(A_continuous).dot(A[i] - np.eye(n)).dot(B_continuous)
            ############################################

            x[i + 1] = A[i].dot(x[i]) + B[i].dot(u[i])
    else:
        ### VARIANT 2:
        # Explicit Solution Formula (current implementation only yields position time series!)
        if final_vel is None:
            final_vel = np.zeros((dim,))
        if final_acc is None:
            final_acc = np.zeros((dim,))
        final_vel = np.array(final_vel)
        final_acc = np.array(final_acc)
        assert final_vel.shape == (dim,)
        assert final_acc.shape == (dim,)
        t_f = passage_times[1] * dt
        coeff_vec = np.array([[x0[0 + i], t_f * x0[1*dim + i], 0.5 * (t_f ** 2) * x0[2*dim + i],
                           -10 * x0[0 + i] - 6 * t_f * x0[1*dim + i] - 1.5 * (t_f ** 2) * x0[2*dim + i] + 10 * T[i] - 4 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i],
                           15 * x0[0 + i] + 8 * t_f * x0[1*dim + i] + 1.5 * (t_f ** 2) * x0[2*dim + i] - 15 * T[i] + 7 * t_f * final_vel[i] - 1 * (t_f ** 2) * final_acc[i],
                           -6 * x0[0 + i] - 3 * t_f * x0[1*dim + i] - 0.5 * (t_f ** 2) * x0[2*dim + i] + 6 * T[i] - 3 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i]] for i in range(dim)] +
                        [[x0[1*dim + i], t_f * x0[2*dim + i],
                           (3 / t_f) * (-10 * x0[0 + i] - 6 * t_f * x0[1*dim + i] - 1.5 * (t_f ** 2) * x0[2*dim + i] + 10 * T[i] - 4 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i]),
                           (4 / t_f) * (15 * x0[0 + i] + 8 * t_f * x0[1*dim + i] + 1.5 * (t_f ** 2) * x0[2*dim + i] - 15 * T[i] + 7 * t_f * final_vel[i] - 1 * (t_f ** 2) * final_acc[i]),
                           (5 / t_f) * (-6 * x0[0 + i] - 3 * t_f * x0[1*dim + i] - 0.5 * (t_f ** 2) * x0[2*dim + i] + 6 * T[i] - 3 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i]), 0] for i in range(dim)] +
                        [[x0[2*dim + i],
                       (2 / t_f) * (3 / t_f) * (-10 * x0[0 + i] - 6 * t_f * x0[1*dim + i] - 1.5 * (t_f ** 2) * x0[2*dim + i] + 10 * T[i] - 4 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i]),
                       (3 / t_f) * (4 / t_f) * (15 * x0[0 + i] + 8 * t_f * x0[1*dim + i] + 1.5 * (t_f ** 2) * x0[2*dim + i] - 15 * T[i] + 7 * t_f * final_vel[i] - 1 * (t_f ** 2) * final_acc[i]),
                       (4 / t_f) * (5 / t_f) * (-6 * x0[0 + i] - 3 * t_f * x0[1*dim + i] - 0.5 * (t_f ** 2) * x0[2*dim + i] + 6 * T[i] - 3 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i]), 0, 0] for i in range(dim)])
        x[:N_MJ + 1, :] = np.squeeze([coeff_vec @ np.array([(j / passage_times[1]) ** ii for ii in range(6)]) for j in range(N_MJ + 1)])
        u[:N_MJ] = T


    # 3. Constantly extend trajectory with last position, zero velocity, and zero acceleration
    for i in range(N_MJ, N):  # extend trajectory with constant last value
        x[i + 1] = x[i] * np.repeat(np.array([1, 0, 0]), dim)
        u[i] = T

    return x, u


def lqr(N, dt, x0, dim,
        r=-np.log(5e-3), velocitycosts_weight=5e-2, forcecosts_weight=1e-3,
        mass=1, t_const_1=0.04, t_const_2=0.04,
        num_targets=1):
    """
    Computes LQR trajectory of given duration, using stage costs and dynamics as described in the paper.
    :param N: number of time steps (excluding final step) [int]
    :param dt: time step duration in seconds [h in paper] [int/float]
    :param x0: initial state ((flattened) position, velocity, muscle force, muscle excitation, target position(s)) [array-like (1D)]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param r: negative log (!) of effort cost weight [corresponds to -np.log(omega_r) with omega_r from paper] [int/float]
    :param velocitycosts_weight: velocity cost weight [omega_v in paper] [positive int/float]
    :param forcecosts_weight: force cost weight [omega_f in paper] [positive int/float]
    :param mass: mass of object to which forces are applied [positive int/float]
    :param t_const_1: time activation constant [tau_1 in paper] [float]
    :param t_const_2: time excitation constant [tau_2 in paper] [float]
    :param num_targets: number of targets (can be used for via-point tasks) [int]
    :return:
        - J: optimal costs [float]
        - x: optimal state sequence [(N+1)*n-array]
        - u: optimal control sequence [N*dim-array]
    """

    # 1. Compute required system matrices
    n = dim * (4 + num_targets)  # dimension of state vector (incorporating position, velocity, force, excitation, and target(s) T (no initial position!))
    Delta = 0  # number of delay time steps
    m = dim * (4 + Delta + num_targets)  # dimension of information vector (corresponding to state vector)
    A = np.vstack((np.hstack((np.eye(dim), dt * np.eye(dim), np.zeros(shape=(dim, n - 2 * dim)))),
                   np.hstack((
                       np.zeros(shape=(dim, dim)), np.eye(dim), (dt / mass) * np.eye(dim),
                       np.zeros(shape=(dim, n - 3 * dim)))),
                   np.hstack((np.zeros(shape=(dim, 2 * dim)), (1 - (dt / t_const_2)) * np.eye(dim),
                              (dt / t_const_2) * np.eye(dim), np.zeros(shape=(dim, n - 4 * dim)))),
                   np.hstack((np.zeros(shape=(dim, 3 * dim)), (1 - (dt / t_const_1)) * np.eye(dim),
                              np.zeros(shape=(dim, n - 4 * dim)))),
                   np.hstack((np.zeros(shape=(n - 4 * dim, 4 * dim)), np.eye(n - 4 * dim)))))
    B = np.vstack(
        (np.zeros(shape=(3 * dim, dim)), (dt / t_const_1) * np.eye(dim), np.zeros(shape=(n - 4 * dim, dim))))
    assert len(x0) == n, "Initial state x0 has wrong dimension!"


    Q = []
    for i in range(1, num_targets + 1):
        D0i = np.vstack((np.hstack((np.eye(dim), np.zeros(shape=(dim, dim * (3 + (i - 1)))), -1 * np.eye(dim),
                                    np.zeros(shape=(dim, n - dim * (4 + i))))),
                         np.zeros(shape=(dim * (3 + (i - 1)), n)),
                         np.hstack((-1 * np.eye(dim), np.zeros(shape=(dim, dim * (3 + (i - 1)))), np.eye(dim),
                                    np.zeros(shape=(dim, n - dim * (4 + i))))),
                         np.zeros(shape=(n - dim * (4 + i), n))))
        assert num_targets == 1
        for j in range(N + 1):
            Q.append(D0i.copy())

        # INFO: Here, velocity and force costs are applied curing the complete movement as well!
        D_v = np.zeros((dim, n))
        D_v[:, dim:(2 * dim)] = np.eye(dim)
        D_f = np.zeros((dim, n))
        D_f[:, (2 * dim):(3 * dim)] = np.eye(dim)
        for i in range(N + 1):
            Q[i] += velocitycosts_weight * (D_v.transpose().dot(D_v))
            Q[i] += forcecosts_weight * (D_f.transpose().dot(D_f))

    R = (np.exp(-r) / (N - 1)) * np.eye(dim)

    # 2. Compute Matrices required for Optimal Control Law
    Ahelp = np.dot((np.linalg.matrix_power(A, Delta)),
                   np.hstack((np.eye(n), np.zeros(shape=(n, m - n + (int(not (m - n))) * dim)))))
    Bhelp = [np.zeros(shape=(n, m + (int(not (m - n))) * dim))] * (Delta + 1)
    for i in range(1, Delta + 1):
        Bhelp[i] = np.dot((np.linalg.matrix_power(A, (i - 1))), B).dot(np.hstack((np.zeros(
            shape=(dim, m - (1 - (int(not (m - n)))) * dim * i)), np.eye(dim), np.zeros(shape=(dim, dim * i - dim)))))
    M = Ahelp + np.sum(Bhelp, 0)

    # WARNING! Not usable for non-constant targets!
    Acal = np.vstack((np.hstack((A, B.dot(np.eye(dim, min(dim, m - n))),
                                 np.zeros(shape=(n, m - n - min(dim, m - n) + (int(not (m - n))) * dim)))),
                      np.hstack((np.zeros(
                          shape=(m - n - min(dim, m - n), n + min(dim, m - n) + (int(not (m - n))) * dim)),
                                 np.eye(m - n - min(dim, m - n)))),
                      np.zeros(shape=(dim, m + (int(not (m - n))) * dim))))
    Bcal = np.vstack((B[:(int(not (m - n))) * n, :], np.zeros(shape=((1 - (int(not (m - n)))) * n, dim)),
                      np.zeros(shape=(m - n - min(dim, m - n), dim)), np.eye(dim)))
    # NOTE: if Delta!=0, i.e., m!=n, B is included in Acal, NOT in Bcal!

    # 3. Compute Optimal Control Law
    #RI1 = R.dot(np.hstack((np.zeros(shape=(dim, m - (1 - (int(not (m - n)))) * dim)), np.eye(dim))))  #only required when penalizing derivative of controls
    #I1RI1 = np.vstack((np.zeros(shape=(m - (1 - (int(not (m - n)))) * dim, dim)), np.eye(dim))).dot(RI1)  #only required when penalizing derivative of controls
    RI1 = np.zeros(shape=(dim, m + dim))
    I1RI1 = np.zeros(shape=(m + dim, m + dim))

    L = [None] * (N + 1)
    L[N] = np.hstack((np.zeros(shape=(dim, m - (1 - (int(not (m - n)))) * dim)), np.eye(dim)))
    S = M.transpose().dot(Q[N]).dot(M)

    for i in range(N - 1, -1, -1):
        Bcal_Siplus1 = Bcal.transpose().dot(S)
        Shelp = RI1 - Bcal_Siplus1.dot(Acal)
        try:
            L[i] = np.linalg.solve(R + Bcal_Siplus1.dot(Bcal), Shelp)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('ERROR in Discrete Riccati Equation: Singular matrix!\n')
                # L[i] = np.linalg.pinv(Bcal_Siplus1.dot(Bcal)).dot(Shelp)
            else:
                raise ValueError
        S = Acal.transpose().dot(S).dot(Acal) + M.transpose().dot(Q[i]).dot(M) + I1RI1 - Shelp.transpose().dot(L[i])


    # 4. (Online) Control Algorithm
    Ical = [None] * (N + 1)
    Ical[0] = np.concatenate((x0, np.zeros((dim, ))))  # extend to information vector space (albeit not required here)
    J = Ical[0].transpose().dot(S).dot(Ical[0])  # optimal (expected) cost

    x = np.zeros((N + 1, n))
    u = np.zeros((N, dim))
    J_cp = 0
    for i in range(0, N):
        u[i] = (L[i].dot(Ical[i]))
        Ical[i + 1] = Acal.dot(Ical[i]) + Bcal.dot(u[i])
        x[i] = (M.dot(Ical[i]))
        J_cp += x[i].transpose().dot(Q[i]).dot(x[i])
        J_cp += (u[i]).transpose().dot(R).dot(u[i])
    x[N] = (M.dot(Ical[N]))
    J_cp += x[N].transpose().dot(Q[N]).dot(x[N])

    if J < 0:
        print(f"!!! IMPORTANT WARNING (LQR): J = {J}!")

    #print(f'LQR - J (expected; backward computation): {J}')
    #print(f'LQR - J (expected; forward computation): {J_cp}')

    return J, x, u


def lqg(N, dt, x0_mean, u0, Sigma0, dim, num_targets,
        r=-np.log(5e-3), velocitycosts_weight=5e-2, forcecosts_weight=1e-3,
        mass=1, t_const_1=0.4, t_const_2=0.4,
        sigma_u=0.2, sigma_c=5e-5,
        sigma_s=0.2, #only used for LQG
        sigma_H=0.01, sigma_Hdot=0.1, sigma_frc=0.5, sigma_e=0.1, gamma=0.5,  #only used for E-LQG
        passage_times=None,
        saccade_times=None,  #only used for E-LQG
        Delta=0,
        system_dynamics="LQG",
        use_information_vector=False,
        include_proprioceptive_target_signals=False,  #only used for LQG
        include_proprioceptive_endeffector_signals=False,  #only used for E-LQG
        modify_init_target_estimate=True,  #only used for E-LQG
        use_square_root_filter=True, minimum_computations=False,
        min_algorithm_iterations=1, algorithm_iterations=20, J_eps=1e-3):
    """
    Computes LQG or E-LQG trajectory of given duration, using stage costs and dynamics as described in the paper.
    :param N: number of time steps (excluding final step) [int]
    :param dt: time step duration in seconds [h in paper] [int/float]
    :param x0_mean: initial state expectation ((flattened) position, velocity, muscle force, muscle excitation, [initial position (only if system_dynamics=="E-LQG"),] target position(s)) [array-like (1D)]
    :param u0: initial control value (only used if "use_information_vector == True") [dim-array]
    :param Sigma0: initial state covariance matrix [array-like (2D)]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param num_targets: number of targets [if system_dynamics=="E-LQG": including initial position] (can be used for via-point tasks) [int]
    :param r: negative log (!) of effort cost weight [corresponds to -np.log(omega_r) with omega_r from paper] [int/float]
    :param velocitycosts_weight: velocity cost weight [omega_v in paper] [positive int/float]
    :param forcecosts_weight: force cost weight [omega_f in paper] [positive int/float]
    :param mass: mass of object to which forces are applied [positive int/float]
    :param t_const_1: time activation constant [tau_1 in paper] [float]
    :param t_const_2: time excitation constant [tau_2 in paper] [float]
    :param sigma_u: signal-dependent (multiplicative) control noise level [float]
    :param sigma_c: constant (i.e., signal-independent) control noise level [float]
    :param sigma_s: observation noise scaling parameter (only used for LQG) [float]
    :param sigma_H: proprioceptive position noise level (only used for E-LQG, only used if "include_proprioceptive_endeffector_signals==True") [float]
    :param sigma_Hdot: visual velocity noise level [sigma_v in paper] (only used for E-LQG) [float]
    :param sigma_frc: visual force noise level [sigma_f in paper] (only used for E-LQG) [float]
    :param sigma_e: gaze noise level (only used for E-LQG) [float]
    :param gamma: visual position noise weight (only used for E-LQG) [float]
    :param passage_times: array of indices that correspond to target passing times in via-point tasks;
    at these time steps, distance, velocity, and force costs are applied [num_targets-array]
    :param saccade_times: array of indices that correspond to saccade times [see n_s in paper] (only used for E-LQG) [num_targets-array]
    :param Delta: observation time lag in time steps (experimental!; default: 0) [int]
    :param system_dynamics: which dynamics to use ["LQG" or "E-LQG"]
    :param use_information_vector: whether to augment state vectors with latest controls (needs to be True if Delta > 0) [bool]
    :param include_proprioceptive_target_signals: whether target position(s) can be observed in absolute coordinates
    (only usable for LQG; default: False) [bool]
    :param include_proprioceptive_endeffector_signals: whether end-effector can be observed position in absolute coordinates
    (only usable for E-LQG; default: False) [bool]
    :param modify_init_target_estimate: whether to place an incorrect initial target estimate
    (basically, the target estimate is set to initial movement position, which makes particular sense for via-point tasks)
    (only works for E-LQG or if "include_proprioceptive_target_signals==True") [bool]
    :param use_square_root_filter: whether to use the square root filter to update Kalman matrices (default: True) [bool]
    :param minimum_computations: if True, realized costs and other stuff are not computed and printed (useful in optimizations etc.) [bool]
    :param min_algorithm_iterations: minimum number of iterations (see "algorithm_iterations")
    :param algorithm_iterations: (maximum) number of iterations, where the optimal control and
    the optimal observation problem is solved alternately (if "J_eps" is set, early termination is possible) [int]
    :param J_eps: if relative improvement of cost function falls below "J_eps" and "min_algorithm_iterations" is reached,
    iterative solving algorithm terminates [float]
    :return:
        - J: optimal (expected) costs [float]
        - x_expectation: mean of expected optimal state sequence (except for target components,
          which correspond to internal estimates at same time) [(N+1)*m-array]
        - Sigma_x: variance of expected optimal (true) state sequence [list of (N+Delta+1) m*m-arrays]
        - u_expectation: expected optimal control sequence [N*dim-array]
        - Ical: sequence internal state estimate means [list of (N+1) m-arrays]
        - x: true state sequence [(N+1)*n-array]
        - u: optimal control sequence (depending on realization of observations y) [N*dim-array]
        - y: observation sequence [(N+1)*l-array]
        - L: optimal feedback gain matrices [list of N dim*m-arrays]
        - Kk: Kalman filter gain matrices [list of N m*l-arrays]
    """

    assert algorithm_iterations >= 0

    if Delta != 0:
        print("WARNING: The current costs cannot be used with Delta > 0. Setting Delta to 0!")
        Delta = 0
        # INFO: if costs are manually provided that allow for Delta > 0, use_information_vector also needs to be set to True!


    # 1. Set up optimal control problem and initialize values
    system_matrices, n, m, l, gamma, passage_times, saccade_times = lqg_initialization(N, dt, dim, r=r,
                                                                                      velocitycosts_weight=velocitycosts_weight,
                                                                                      forcecosts_weight=forcecosts_weight,
                                                                                      mass=mass,
                                                                                      t_const_1=t_const_1,
                                                                                      t_const_2=t_const_2,
                                                                                      sigma_u=sigma_u,
                                                                                      sigma_c=sigma_c,
                                                                                      sigma_H=sigma_H,
                                                                                      sigma_Hdot=sigma_Hdot,
                                                                                      sigma_e=sigma_e,
                                                                                      gamma=gamma,
                                                                                      sigma_s=sigma_s,
                                                                                      sigma_frc=sigma_frc,
                                                                                      passage_times=passage_times,
                                                                                      saccade_times=saccade_times,
                                                                                      Delta=Delta,
                                                                                      system_dynamics=system_dynamics,
                                                                                      num_targets=num_targets,
                                                                                      use_information_vector=use_information_vector,
                                                                                      include_proprioceptive_endeffector_signals=include_proprioceptive_endeffector_signals,
                                                                                      include_proprioceptive_target_signals=include_proprioceptive_target_signals)

    initial_state_matrices = lqg_setup_initial_state(dim, num_targets, x0_mean, u0, Sigma0, n, m, Delta=Delta,
                                                     system_dynamics=system_dynamics,
                                                     use_information_vector=use_information_vector,
                                                     include_proprioceptive_target_signals=include_proprioceptive_target_signals,
                                                     modify_init_target_estimate=modify_init_target_estimate)
    Ical0, Sigma_x0 = initial_state_matrices[:2]
    J = 10e+12  #use very large costs costs as initial reference value
    Kk = None


    # 2. Alternately solve the optimal control given fixed Kalman gain matrices Kk and the optimal observation problem given fixed feedback gains L
    for iter in range(algorithm_iterations + 1):

        # 2.1. Solve the optimal control problem given fixed Kalman gain matrices Kk and compute resulting costs
        J_before = J
        L, S_x, S_e, alpha = lqg_solve_control_problem(N, dim, num_targets, gamma, saccade_times, system_matrices, m, l,
                                                   Delta=Delta, Kk=Kk)
        J = lqg_compute_optimal_costs(Ical0, Sigma_x0, S_x, S_e, alpha)
        # assert J >= 0
        if J < 0:
            print(f"!!! IMPORTANT WARNING (LQG): J = {J}!")


        # 2.2. Solve the optimal observation problem given fixed feedback gains L
        Kk, Kcal, Sigma_Ical, Sigma_ecal, Sigma_Icalecal = lqg_solve_observation_problem(N, dim, num_targets,
                                                                                         gamma,
                                                                                         saccade_times,
                                                                                         system_matrices,
                                                                                         initial_state_matrices,
                                                                                         m, l, Delta=Delta, L=L,
                                                                                         use_square_root_filter=use_square_root_filter)


        # 2.3. Decide whether to stop iteration
        if J_eps is not None:
            if (J <= J_before) and (np.abs((J - J_before)/J_before) <= J_eps) and (iter >= min_algorithm_iterations - 1):
                if not minimum_computations:
                    print(f'{system_dynamics} converged after {iter + 1} iterations.')
                break
            elif iter == algorithm_iterations - 1:
                if not minimum_computations:
                    print(f'WARNING: LQG stopped after {iter + 1} iterations without convergence.')

    # 3. Run forward pass using optimal feedback gains L and Kalman gains Kk
    J_cp, J_sample, x_expectation, Sigma_x, u_expectation, Ical, x, u, y = lqg_forward_simulation(
        L,
        Kk, Kcal, Sigma_Ical, Sigma_ecal, Sigma_Icalecal,
        N,
        dim,
        num_targets,
        x0_mean, u0, Sigma0,
        system_matrices,
        initial_state_matrices,
        n, m, l,
        gamma, saccade_times,
        Delta=Delta,
        minimum_computations=minimum_computations,
        use_information_vector=use_information_vector)

    if not minimum_computations:
        print(f'LQG - J (expected; backward computation): {J}')
        print(f'LQG - J (sample-based (internal); forward computation): {J_cp}')
        print(f'LQG - J (sample-based; forward computation): {J_sample}')
        # print(f'LQG - J (sample-based; forward computation; ONLY STATE COSTS): {J_sample_state}')

    return J, x_expectation, Sigma_x, u_expectation, Ical, x, u, y, L, Kk

def lqg_initialization(N, dt, dim,
                       r=-np.log(5e-7), velocitycosts_weight=1.0, forcecosts_weight=1.0,
                       mass=1, t_const_1=0.4, t_const_2=0.4,
                       sigma_u=0.2, sigma_c=5e-5, sigma_s=0.2, sigma_H=0.01, sigma_Hdot=0.1, sigma_frc=0.5, sigma_e=0.1, gamma=0.5,
                       passage_times=None, saccade_times=None, Delta=0,
                       system_dynamics="LQG",
                       num_targets=1,
                       use_information_vector=False,
                       include_proprioceptive_target_signals=False,
                       include_proprioceptive_endeffector_signals=False):
    """
    Provides system matrices required to compute LQG/E-LQG solution.
    :param N: number of time steps (excluding final step) [int]
    :param dt: time step duration in seconds [h in paper] [int/float]
    :param x0_mean: initial state expectation ((flattened) position, velocity, muscle force, muscle excitation, [initial position (only if system_dynamics=="E-LQG"),] target position(s)) [array-like (1D)]
    :param u0: initial control value (only used if "use_information_vector == True") [dim-array]
    :param Sigma0: initial state covariance matrix [array-like (2D)]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param r: negative log (!) of effort cost weight [corresponds to -np.log(omega_r) with omega_r from paper] [int/float]
    :param velocitycosts_weight: velocity cost weight [omega_v in paper] [positive int/float]
    :param forcecosts_weight: force cost weight [omega_f in paper] [positive int/float]
    :param mass: mass of object to which forces are applied [positive int/float]
    :param t_const_1: time activation constant [tau_1 in paper] [float]
    :param t_const_2: time excitation constant [tau_2 in paper] [float]
    :param sigma_u: signal-dependent (multiplicative) control noise level [float]
    :param sigma_c: constant (i.e., signal-independent) control noise level [float]
    :param sigma_s: observation noise scaling parameter (only used for LQG) [float]
    :param sigma_H: proprioceptive position noise level (only used for E-LQG, only used if "include_proprioceptive_endeffector_signals==True") [float]
    :param sigma_Hdot: visual velocity noise level [sigma_v in paper] (only used for E-LQG) [float]
    :param sigma_frc: visual force noise level [sigma_f in paper] (only used for E-LQG) [float]
    :param sigma_e: gaze noise level (only used for E-LQG) [float]
    :param gamma: visual position noise weight (only used for E-LQG) [float]
    :param passage_times: array of indices that correspond to target passing times in via-point tasks;
    at these time steps, distance, velocity, and force costs are applied [num_targets-array]
    :param saccade_times: array of indices that correspond to saccade times [see n_s in paper] (only used for E-LQG) [num_targets-array]
    :param Delta: observation time lag in time steps (experimental!; default: 0) [int]
    :param system_dynamics: which dynamics to use ["LQG" or "E-LQG"]
    :param num_targets: number of targets [if system_dynamics=="E-LQG": including initial position] (can be used for via-point tasks) [int]
    :param use_information_vector: whether to augment state vectors with latest controls (needs to be True if Delta > 0) [bool]
    :param include_proprioceptive_target_signals: whether target position(s) can be observed in absolute coordinates
    (only usable for LQG; default: False) [bool]
    :param include_proprioceptive_endeffector_signals: whether end-effector can be observed position in absolute coordinates
    (only usable for E-LQG; default: False) [bool]
    :return:
        - system_matrices: tuple of relevant system matrices (see definition below) that only need to be computed once at the beginning [tuple]
        - n: dimension of state vector (see comment below)
        - m: dimension of information vector (see comment below)
        - l: dimension of observation vector (see comment below)
        - gamma: see above
        - passage_times: see above (might be set to default values if initially None)
        - saccade_times: see above (might be set to default values if initially None)
    """
    # INFO: for E-LQG, proprioceptive noise (defined by sigma_H, sigma_e) should be much larger than visual noise (defined by gamma) (following Todorov1998, p.55)

    if passage_times is None:
        passage_times = np.linspace(0, N, num_targets).astype(int) if num_targets > 1 \
            else np.array([N]) # WARNING: here: equally distributed target passage times!
    if saccade_times is None:
        saccade_times = np.clip(-Delta + 50 + np.linspace(0, N, num_targets), -Delta, -Delta + N).astype(
            int)  # from Todorov1998, Fig. 4-6  #WARNING: optimize over eye fixation times!
    if system_dynamics == "E-LQG":
        assert len(passage_times) == num_targets
        assert len(saccade_times) == num_targets
    Delta = np.round(Delta).astype(int)  # cast Delta to integer
    #gamma = np.max((gamma, 4e-18))  # ensure positive gamma to avoid numerical instabilities


    # 1. Compute system dynamics matrices
    n = dim * (4 + num_targets)  # dimension of state vector (incorporating position, velocity, force, excitation, [initial position (only if system_dynamics=="E-LQG"),] and target(s) T)
    m = dim * (4 + num_targets + max(1, Delta) * use_information_vector)  # dimension of information vector (corresponding to state vector)
    if system_dynamics == "E-LQG":
        l = dim * (3 + include_proprioceptive_endeffector_signals + num_targets)  # dimension of observation vector (incorporating position [if include_proprioceptive_endeffector_signals], velocity, and force [and eye fixation, and targets relative to eye fixation (defined by saccade_times and list of targets)])
    elif system_dynamics == "LQG":
        l = dim * (3 + (num_targets * include_proprioceptive_target_signals))  # dimension of observation vector (incorporating position, velocity, and force [and targets])
    else:
        raise NotImplementedError
    # INFO: If Delta=0, m!=n still holds (i.e., information vector has size dim*(2+num_targets+1)=m+dim), since u_(k-1) needs to be propagated if the derivatives of controls are penalized!
    A = np.vstack((np.hstack((np.eye(dim), dt * np.eye(dim), np.zeros(shape=(dim, n - 2 * dim)))),
                   np.hstack((
                       np.zeros(shape=(dim, dim)), np.eye(dim), (dt/mass) * np.eye(dim), np.zeros(shape=(dim, n - 3 * dim)))),
                   np.hstack((np.zeros(shape=(dim, 2 * dim)), (1 - (dt/t_const_2)) * np.eye(dim),
                        (dt/t_const_2) * np.eye(dim), np.zeros(shape=(dim, n - 4 * dim)))),
                   np.hstack((np.zeros(shape=(dim, 3 * dim)), (1 - (dt/t_const_1)) * np.eye(dim),
                              np.zeros(shape=(dim, n - 4 * dim)))),
                   np.hstack((np.zeros(shape=(n - 4 * dim, 4 * dim)), np.eye(n - 4 * dim)))))
    B = np.vstack((np.zeros(shape=(3 * dim, dim)), (dt/t_const_1) * np.eye(dim), np.zeros(shape=(n - 4 * dim, dim))))

    C = sigma_u * B
    D = sigma_c * np.diag(np.sum(B, axis=1))   # additive noise is only applied to control

    # 2. Compute observation dynamics matrices
    def get_current_Hk__ELQG(index, saccade_times, dim, num_targets):
        """
        Provides current observation matrix.
        Here, observation model from Todorov98_thesis is combined with system dynamics (LQG) from Todorov05.
        :param index: current time step [int]
        :param saccade_times: array of indices that correspond to saccade times [see n_s in paper] [num_targets-array]
        :param dim: dimension of the task (1D, 2D, or 3D) [int]
        :param num_targets: number of targets including initial position (can be used for via-point tasks) [int]
        :return:
            - Hk: observation matrix at current time step [l*n-array]
            - f_k: index of targets that is currently fixated (starting with 1 as initial position, 2 as first target, ...) [int]
        """
        remaining_fixations_indices = [i + 1 for i, switch_time in enumerate(saccade_times) if index <= switch_time]
        remaining_fixations_indices_next = [i + 1 for i, switch_time in enumerate(saccade_times) if
                                            index + 1 <= switch_time]

        if len(remaining_fixations_indices) > 0:
            f_k = remaining_fixations_indices[0]
        else:  # stay at last fixation point (= last target) if there are some time steps remaining...
            f_k = len(saccade_times)
        if include_proprioceptive_endeffector_signals:
            Hk = np.vstack((np.hstack((np.eye(3 * dim), np.zeros(shape=(3 * dim, n - 3 * dim)))),
                            np.hstack((np.zeros(shape=(dim, n + (-num_targets + f_k - 1) * dim)), np.eye(dim),
                                       np.zeros(shape=(dim, (num_targets - f_k) * dim)))),
                            np.hstack(
                                (np.eye(dim), np.zeros(shape=(dim, n + (-num_targets + f_k - 2) * dim)), -np.eye(dim),
                                 np.zeros(shape=(dim, (num_targets - f_k) * dim)))),
                            np.delete(np.hstack(
                                (np.zeros((num_targets * dim, n - num_targets * dim)), np.eye(num_targets * dim))) +
                                      np.hstack(
                                          (np.zeros(shape=(num_targets * dim, n + (-num_targets + f_k - 1) * dim)),
                                           np.tile(-np.eye(dim), (num_targets, 1)),
                                           np.zeros(shape=(num_targets * dim, (num_targets - f_k) * dim)))),
                                      obj=range((f_k - 1) * dim, f_k * dim), axis=0)
                            ))
        else:
            Hk = np.vstack(
                (np.hstack((np.zeros(shape=(2 * dim, dim)), np.eye(2 * dim), np.zeros(shape=(2 * dim, n - 3 * dim)))),
                 np.hstack((np.zeros(shape=(dim, n + (-num_targets + f_k - 1) * dim)), np.eye(dim),
                            np.zeros(shape=(dim, (num_targets - f_k) * dim)))),
                 np.hstack(
                     (np.eye(dim), np.zeros(shape=(dim, n + (-num_targets + f_k - 2) * dim)), -np.eye(dim),
                      np.zeros(shape=(dim, (num_targets - f_k) * dim)))),
                 np.delete(np.hstack(
                     (np.zeros((num_targets * dim, n - num_targets * dim)), np.eye(num_targets * dim))) +
                           np.hstack(
                               (np.zeros(shape=(num_targets * dim, n + (-num_targets + f_k - 1) * dim)),
                                np.tile(-np.eye(dim), (num_targets, 1)),
                                np.zeros(shape=(num_targets * dim, (num_targets - f_k) * dim)))),
                           obj=range((f_k - 1) * dim, f_k * dim), axis=0)
                 ))
        if (len(remaining_fixations_indices_next) < len(remaining_fixations_indices)) and ((saccade_times[
                                                                                                f_k - 1] % 1) != 0):  # switch time saccade_times[f_k - 1] is between index and index + 1  --> interpolate between Hk's
            if len(remaining_fixations_indices_next) > 0:
                f_k_next = remaining_fixations_indices_next[0]
            else:  # stay at last fixation point (= last target) if there are some time steps remaining...
                f_k_next = len(saccade_times)
            if include_proprioceptive_endeffector_signals:
                Hk_next = np.vstack((np.hstack((np.eye(3 * dim), np.zeros(shape=(3 * dim, n - 3 * dim)))),
                                     np.hstack(
                                         (np.zeros(shape=(dim, n + (-num_targets + f_k_next - 1) * dim)), np.eye(dim),
                                          np.zeros(shape=(dim, (num_targets - f_k_next) * dim)))),
                                     np.hstack((np.eye(dim),
                                                np.zeros(shape=(dim, n + (-num_targets + f_k_next - 2) * dim)),
                                                -np.eye(dim),
                                                np.zeros(shape=(dim, (num_targets - f_k_next) * dim)))),
                                     np.delete(
                                         np.hstack((np.zeros((num_targets * dim, n - num_targets * dim)),
                                                    np.eye(num_targets * dim))) +
                                         np.hstack((np.zeros(
                                             shape=(num_targets * dim, n + (-num_targets + f_k_next - 1) * dim)),
                                                    np.tile(-np.eye(dim), (num_targets, 1)),
                                                    np.zeros(
                                                        shape=(num_targets * dim, (num_targets - f_k_next) * dim)))),
                                         obj=range((f_k_next - 1) * dim, f_k_next * dim), axis=0)
                                     ))
            else:
                Hk_next = np.vstack((np.hstack(
                    (np.zeros(shape=(2 * dim, dim)), np.eye(2 * dim), np.zeros(shape=(2 * dim, n - 3 * dim)))),
                                     np.hstack(
                                         (np.zeros(shape=(dim, n + (-num_targets + f_k_next - 1) * dim)), np.eye(dim),
                                          np.zeros(shape=(dim, (num_targets - f_k_next) * dim)))),
                                     np.hstack((np.eye(dim),
                                                np.zeros(shape=(dim, n + (-num_targets + f_k_next - 2) * dim)),
                                                -np.eye(dim),
                                                np.zeros(shape=(dim, (num_targets - f_k_next) * dim)))),
                                     np.delete(
                                         np.hstack((np.zeros((num_targets * dim, n - num_targets * dim)),
                                                    np.eye(num_targets * dim))) +
                                         np.hstack((np.zeros(
                                             shape=(num_targets * dim, n + (-num_targets + f_k_next - 1) * dim)),
                                                    np.tile(-np.eye(dim), (num_targets, 1)),
                                                    np.zeros(
                                                        shape=(num_targets * dim, (num_targets - f_k_next) * dim)))),
                                         obj=range((f_k_next - 1) * dim, f_k_next * dim), axis=0)
                                     ))
            Hk = (saccade_times[f_k - 1] % 1) * Hk + (1 - (saccade_times[f_k - 1] % 1)) * Hk_next
            f_k = (saccade_times[f_k - 1] % 1) * f_k + (1 - (saccade_times[f_k - 1] % 1)) * f_k_next
            if f_k == int(
                    f_k):  # should only be the case if len(remaining_fixations_indices_next) == 0, i.e., f_k == f_k_next
                f_k = int(f_k)

        return Hk, f_k


    #  Otherwise, try to use get_linearized_noise_terms() instead, which linearizes noise terms around MinJerk trajectory...
    if system_dynamics in ["LQG"]:
        Gk_constant = np.diag(sigma_s * np.array([0.02, 0.2, 1] + ([0.02] * include_proprioceptive_target_signals)).repeat(dim))
    elif system_dynamics in ["E-LQG"]:
        if include_proprioceptive_endeffector_signals:
            Gk_constant = np.diag(np.hstack((np.array([sigma_H, sigma_Hdot, sigma_frc, sigma_e]),
                                             np.zeros(shape=(num_targets,)))).repeat(dim))
        else:
            Gk_constant = np.diag(np.hstack((np.array([sigma_Hdot, sigma_frc, sigma_e]),
                                             np.zeros(shape=(num_targets,)))).repeat(dim))
    else:
        if include_proprioceptive_endeffector_signals:
            Gk_constant = np.diag(np.hstack((np.array([sigma_H, sigma_Hdot, sigma_e]),
                                             np.zeros(shape=(num_targets,)))).repeat(dim))
        else:
            Gk_constant = np.diag(np.hstack((np.array([sigma_Hdot, sigma_e]),
                                             np.zeros(shape=(num_targets,)))).repeat(dim))

    def get_current_Gk__ELQG(dim, num_targets, Hk, gamma):
        """
        Provides list of internally used observation noise submatrices, given the corresponding observation matrix Hk.
        Here, observation model from Todorov98_thesis is combined with system dynamics (LQG) from Todorov05.
        :param dim: dimension of the task (1D, 2D, or 3D) [int]
        :param num_targets: number of targets including initial position (can be used for via-point tasks) [int]
        :param Hk: observation matrix [l*n-array]
        :return: list of internally used observation noise submatrices [list of (l/dim) l*n-arrays]
        """
        if dim > 1:
            assert gamma == 0, "ERROR: These state-dependent observation noise matrices cannot be used for dim > 1! Set gamma=0 or use dim=1."

        Gk = []
        Gk_coeff = np.diag(np.hstack((np.zeros(shape=(3 + include_proprioceptive_endeffector_signals,)),
                                      np.sqrt(gamma) * np.ones(shape=(num_targets,)))).repeat(dim))
        for ii in range(3 + include_proprioceptive_endeffector_signals + num_targets):
            Gk.append(Gk_coeff.dot(
                np.diag(np.hstack((np.zeros(ii * dim, ), np.ones(dim, ), np.zeros(l - (ii + 1) * dim, ))))).dot(Hk))
        return Gk

    if system_dynamics == "LQG":  # following Todorov05
        get_current_Gk = lambda dim, var1, var2, var3: [np.zeros(shape=(l, n))]
        Hk = np.vstack((np.hstack((np.eye(3 * dim), np.zeros(shape=(3 * dim, n - 3 * dim)))),
                        np.hstack((np.zeros(shape=((num_targets * include_proprioceptive_target_signals) * dim,
                                                   n - (num_targets * include_proprioceptive_target_signals) * dim)),
                                   np.eye((num_targets * include_proprioceptive_target_signals) * dim)))))
        get_current_Hk = lambda var1, var2, var3, var4: (Hk, np.nan)
    elif system_dynamics == "E-LQG":  # observation model from Todorov98_thesis, with system dynamics from Todorov05 (E-LQG)
        get_current_Gk = get_current_Gk__ELQG
        get_current_Hk = get_current_Hk__ELQG

    # currently unused, but might be a suitable alternative for dim > 1 (requires a different implementation using S and P instead of S_x and S_e [see Todorov98_thesis]!):
    def get_linearized_noise_terms(j, saccade_times, dim, num_targets, X_bar, Sigma_bar, S, P):
        Hk_helper, f_k_helper = get_current_Hk(j, saccade_times, dim, num_targets)
        if int(f_k_helper) != f_k_helper:
            L_X, L_Sigma, g_bar = _get_linearized_noise_terms(j, np.floor(saccade_times), dim, num_targets, X_bar,
                                                              Sigma_bar, S, P)
            L_X_next, L_Sigma_next, g_bar_next = _get_linearized_noise_terms(j + 1, np.floor(saccade_times), dim,
                                                                             num_targets, X_bar, Sigma_bar, S, P)
            L_X = (f_k_helper % 1) * L_X + (1 - (f_k_helper % 1)) * L_X_next
            L_Sigma = (f_k_helper % 1) * L_Sigma + (1 - (f_k_helper % 1)) * L_Sigma_next
            g_bar = (f_k_helper % 1) * g_bar + (1 - (f_k_helper % 1)) * g_bar_next
        else:
            L_X, L_Sigma, g_bar = _get_linearized_noise_terms(j, saccade_times, dim, num_targets, X_bar, Sigma_bar, S,
                                                              P)
        return L_X, L_Sigma, g_bar
    def _get_linearized_noise_terms(j, saccade_times, dim, num_targets, X_bar, Sigma_bar, S, P):
        # TODO: check indices
        Hk_helper, f_k_helper = get_current_Hk(j, saccade_times, dim, num_targets)
        Hk_times_x = Hk_helper.dot(X_bar[j])
        if include_proprioceptive_endeffector_signals:
            Fk_helper = np.diag(np.hstack((np.array([sigma_H ** 2, sigma_Hdot ** 2, sigma_e ** 2]),
                                           gamma * np.linalg.norm(Hk_times_x[dim * 3:].reshape(-1, dim),
                                                                  axis=1) ** 2)).repeat(dim))
        else:
            Fk_helper = np.diag(np.hstack((np.array([sigma_Hdot ** 2, sigma_e ** 2]),
                                           gamma * np.linalg.norm(Hk_times_x[dim * 2:].reshape(-1, dim),
                                                                  axis=1) ** 2)).repeat(dim))
        hlp_hlp = Hk_helper.dot(Sigma_bar[j]).dot(Hk_helper.transpose()) + Fk_helper
        hlp_hlp[np.abs(hlp_hlp) < 10e-10] = 0
        try:
            helper_2 = np.linalg.inv(hlp_hlp)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                # print(f"WARNING: Cannot invert observation variance matrix [{j}] used to compute linearized noise terms, as it is singular. Using pseudo-inverse to proceed.")
                helper_2 = np.linalg.pinv(hlp_hlp)
            else:
                raise np.linalg.LinAlgError
        Z = A.transpose().dot(S[:n, :n] - P).dot(A)

        W = np.diag(-helper_2.dot(Hk_helper.dot(Sigma_bar[j]).dot(Z).dot(Sigma_bar[j]).dot(Hk_helper.transpose())).dot(
            helper_2))[3 * dim:]  # TODO: check dimensions (see Todorov1998, p.48)
        L_X = np.zeros(shape=(n, n))
        for ii in [ii for ii in range(num_targets) if ii != f_k_helper]:
            L_X += gamma * W[ii] * get_Dif_transpose_Dif(ii, f_k_helper)
        if max(W) > 10e+15:
            print(W, j, S, P)
            print('WARNING: W has entries larger than 10e+15!')
            # return ValueError
        T = Hk_helper.transpose().dot(helper_2).dot(Hk_helper)
        L_Sigma = T.dot(Sigma_bar[j]).dot(Z) + Z.dot(Sigma_bar[j]).dot(T) - T.dot(Sigma_bar[j]).dot(Z).dot(
            Sigma_bar[j]).dot(T)
        g = np.trace(Z.dot(Sigma_bar[j]).dot(Hk_helper.transpose()).dot(helper_2).dot(Hk_helper).dot(Sigma_bar[j]))
        g_bar = g - X_bar[j].T.dot(L_X).dot(X_bar[j]) - np.trace(L_Sigma.dot(Sigma_bar[j]))

        return L_X, L_Sigma, g_bar


    # 3. Compute cost matrices
    def get_Dif_transpose_Dif(index_1, index_2):
        """
        Compute square matrix that can be used to quadratically penalize the difference between the state components
        of both indices. Index 0 corresponds to end-effector position, positive indices to respective target
        (starting with 1 as first target (initial position in E-LQG), 2 as second target, ...).
        :param index_1: first state component (positive entry) [int]
        :param index_2: second state component (negative entry) [int]
        :return: Square matrix [n*n-array]
        """
        Dif = np.zeros((dim, n))
        if index_1 == 0:
            Dif[:, index_1 * dim:(index_1 + 1) * dim] = np.eye(dim)
        else:  # skip hand velocity in state vector [at indices "dim:2*dim"]
            Dif[:, n + (-num_targets + index_1 - 1) * dim: n + (-num_targets + index_1) * dim] = np.eye(dim)
        if index_2 == 0:
            Dif[:, index_2 * dim:(index_2 + 1) * dim] = -np.eye(dim)
        else:  # skip hand velocity in state vector [at indices "dim:2*dim"]
            Dif[:, n + (-num_targets + index_2 - 1) * dim: n + (-num_targets + index_2) * dim] = -np.eye(dim)

        return Dif.transpose().dot(Dif)


    D_v = np.zeros((dim, n))
    D_v[:, dim:(2 * dim)] = np.eye(dim)

    D_f = np.zeros((dim, n))
    D_f[:, (2 * dim):(3 * dim)] = np.eye(dim)

    Q = []
    ### Penalize remaining distance to target center only at passage times [Todorov05]
    for i in range(N + 1):
        remaining_targets_indices = [j + 1 for j, switch_time in enumerate(passage_times) if
                                     i <= np.ceil(switch_time)]
        if len(remaining_targets_indices) > 0:
            f_k = remaining_targets_indices[0]
            if i == np.floor(passage_times[f_k - 1]):
                distancecosts_weight = (1 - (passage_times[f_k - 1] % 1))  # * (1 / widthm[0][f_k - 1])
                Q.append(distancecosts_weight * get_Dif_transpose_Dif(0, f_k))
            elif i == np.ceil(passage_times[f_k - 1]):
                distancecosts_weight = (passage_times[f_k - 1] % 1)  # * (1 / widthm[0][f_k - 1])
                Q.append(distancecosts_weight * get_Dif_transpose_Dif(0, f_k))
            else:
                Q.append(np.zeros(shape=(n, n)))
        else:
            Q.append(np.zeros(shape=(n, n)))  # no state costs are given after reaching the (last) target

    ### Apply velocity and force costs only at passage times [Todorov05]
    for passage_time in passage_times:
        # Q[passage_time] += velocitycosts_weight * (D_v.transpose().dot(D_v))
        Q[np.floor(passage_time).astype(int)] += (1 - (passage_time % 1)) * velocitycosts_weight * (
            D_v.transpose().dot(D_v))
        Q[np.ceil(passage_time).astype(int)] += (passage_time % 1) * velocitycosts_weight * (
            D_v.transpose().dot(D_v))
        Q[np.floor(passage_time).astype(int)] += (1 - (passage_time % 1)) * forcecosts_weight * (
            D_f.transpose().dot(D_f))
        Q[np.ceil(passage_time).astype(int)] += (passage_time % 1) * forcecosts_weight * (
            D_f.transpose().dot(D_f))

    ### Continuous control costs
    R = (np.exp(-r) / (N - 1)) * np.eye(dim)


    # 4. Compute helping matrices required for computing the optimal control law
    I_x = np.hstack((np.eye(n), np.zeros(shape=(n, m - n))))
    I_p = [np.zeros(shape=(dim, m))] * (Delta + 1)
    for i in range(1, Delta + 1):  # WARNING: this state_costs_variant is incompatible with Delta > 0!!
        I_p[i] = np.hstack((np.zeros(
            shape=(dim, m - (dim * i))), np.eye(dim), np.zeros(shape=(dim, dim * i - dim))))
    # I_1 = np.hstack((np.zeros(shape=(dim, m - dim)), np.eye(dim)))

    Ahelp = np.dot((np.linalg.matrix_power(A, Delta)), I_x)
    Bhelp = [np.zeros(shape=(n, m))] * (Delta + 1)
    C_p = [np.zeros(shape=(n, m))] * (Delta + 1)
    Dhelp = [np.zeros(shape=(n, n))] * (Delta + 1)
    Shelp = np.zeros(shape=(m, m))

    for i in range(1, Delta + 1):
        Bhelp[i] = np.dot((np.linalg.matrix_power(A, (i - 1))), B).dot(I_p[i])
        C_p[i] = np.dot((np.linalg.matrix_power(A, (i - 1))), C).dot(I_p[i])
        Dhelp_help = np.dot((np.linalg.matrix_power(A, (i - 1))), D)
        Dhelp[i] = np.dot(Dhelp_help, Dhelp_help.T)
        Shelp += np.dot(C_p[i].T, Q[N]).dot(C_p[i])
    M = Ahelp + np.sum(Bhelp, 0)

    # WARNING! Not usable for non-constant targets!
    if not use_information_vector:
        Acal = A
        Bcal = B
        Ccal_1 = np.vstack(
            (np.hstack((np.zeros(shape=(n, n)), sigma_u * B.dot(np.eye(dim, min((Delta > 0) * dim, m - n))),
                        np.zeros(shape=(n, m - n - min((Delta > 0) * dim, m - n))))),
             np.zeros(shape=(m - n, m))))  # only used if Delta > 0
        Ccal_2 = sigma_u * B
        Dcal = D
    else:
        Acal = np.vstack((np.hstack((A, B.dot(np.eye(dim, min((Delta > 0) * dim, m - n))),
                                     np.zeros(shape=(n, m - n - min((Delta > 0) * dim, m - n))))),
                          np.hstack((np.zeros(
                              shape=(m - n - min(dim, m - n), n + min(dim, m - n))),
                                     np.eye(m - n - min(dim, m - n)))),
                          np.zeros(shape=(dim, m))))
        Bcal = np.vstack((B if Delta == 0 else np.zeros(shape=(n, dim)),
                          np.zeros(shape=(max(0, m - n - dim), dim)), np.eye(dim)))
        Ccal_1 = np.vstack((np.hstack((np.zeros(shape=(n, n)), sigma_u * B.dot(np.eye(dim, min((Delta > 0) * dim, m - n))),
                                       np.zeros(shape=(n, m - n - min((Delta > 0) * dim, m - n))))),
                            np.zeros(shape=(m - n, m))))  # only used if Delta > 0
        Ccal_2 = np.vstack((sigma_u * B if Delta == 0 else np.zeros(shape=(n, dim)),
                            np.zeros(shape=(m - n, dim))))  # only used if Delta == 0
        Dcal = np.vstack((D, np.zeros(shape=(m - n, n))))

    # NOTE: if Delta!=0, B is included in Acal, NOT in Bcal!
    Dhelp_sum = np.sum(Dhelp, 0)

    Dcal_help = [np.zeros(shape=(m, m))] * (Delta + 1)
    for i in range(1, Delta + 1):
        Dcal_help_help = np.dot((np.linalg.matrix_power(Acal, (i - 1))), Dcal)
        Dcal_help[i] = np.dot(Dcal_help_help, Dcal_help_help.T)


    system_matrices = (A, B, C, D, Acal, Bcal, Ccal_1, Ccal_2, Dcal, get_current_Hk, get_current_Gk, Gk_constant, Q, R, M, I_p, I_x, C_p, Bhelp, Dhelp, Dhelp_sum, Dcal_help)

    return system_matrices, n, m, l, gamma, passage_times, saccade_times


def lqg_setup_initial_state(dim, num_targets, x0_mean, u0, Sigma0, n, m, Delta=0,
                            system_dynamics="LQG",
                            use_information_vector=False, include_proprioceptive_target_signals=False,
                            modify_init_target_estimate=True):
    """
    Initializes LQG/E-LQG problem by setting up initial state variables.
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param num_targets: number of targets [if system_dynamics=="E-LQG": including initial position] (can be used for via-point tasks) [int]
    :param x0_mean: initial state expectation ((flattened) position, velocity, muscle force, muscle excitation, [initial position (only if system_dynamics=="E-LQG"),] target position(s)) [n-array]
    :param u0: initial control value (only used if "use_information_vector == True") [dim-array]
    :param Sigma0: initial state covariance matrix [n*n-array]
    :param n: dimension of state vector [int]
    :param m: dimension of information vector [int]
    :param Delta: observation time lag in time steps (experimental!; default: 0) [int]
    :param system_dynamics: which dynamics to use ["LQG" or "E-LQG"]
    :param use_information_vector: whether to augment state vectors with latest controls (needs to be True if Delta > 0) [bool]
    :param include_proprioceptive_target_signals: whether target position(s) can be observed in absolute coordinates
    (only usable for LQG; default: False) [bool]
    :param modify_init_target_estimate: whether to place an incorrect initial target estimate
    (basically, the target estimate is set to initial movement position, which makes particular sense for via-point tasks)
    (only works for E-LQG or if "include_proprioceptive_target_signals==True") [bool]
    :return:
        - Ical0: (initial) internal state estimate mean [m-array]
        - Sigma_x0: (initial) internal (true) state covariance [m*m-array]
        - Sigma_ecal0: (initial) covariance of estimation error ecal (ecal = true state (~x) - estimated state mean (~Ical)) [m*m-array]
        - Sigma_Icalecal0: (initial) covariance between internal state estimate Ical and estimation error ecal [m*m-array]
        - Sigma_Ical0: (initial) internal state estimate covariance [m*m-array]
        - Sigma_ecal_sqrt0: square root of Sigma_ecal0 (used for square root filter) [m*m-array]
        - x0_mean_prior_estimate: modified initial state expectation (see below; equals Ical0 if "use_information_vector == False") [n-array]
        - Ical_expectation0: (initial) mean of expected optimal state (or rather internal state estimate) [m-array]
        - ecal_expectation0: (initial) mean of expected estimation error [m-array]
        - Ical_apriori0: (initial) internal state estimate mean of sequence of state estimates with optimal closed-loop control
        (given some Kalman matrices, e.g., via iteration between Kalman and Control Law Matrix updates),
        but open-loop forward pass (i.e., Kalman matrices are not used in system equations) [m-array]
    """
    if u0 is None:
        u0 = np.array([0] * dim)  #necessary for u0.repeat() below, even though u0 is unused in this case

    # Modify initial target estimate (basically, the target estimate is set to initial movement position, which makes even more sense for via-point tasks)
    x0_mean_prior_estimate = x0_mean.copy()  # INFO: x0_mean and Sigma0 corresponds to timestep-Delta
    if ((system_dynamics == "E-LQG") or include_proprioceptive_target_signals) and (
    modify_init_target_estimate):  # otherwise, Todorov05 model (system_dynamics=="LQG") is assumed, where targets are not observable during execution, i.e., true target needs to be given in advance
        if num_targets > 1:
            # Replace initial estimates of (remaining) targets with (true value of) initial "target":
            x0_mean_prior_estimate[n - num_targets * dim:] = np.tile(
                x0_mean_prior_estimate[n - num_targets * dim: n - num_targets * dim + dim], num_targets)
        elif num_targets == 1:  # true initial target position is not part of state vector and thus not available here...
            # Replace initial estimates of (remaining) targets with (estimated, expected) initial position:
            x0_mean_prior_estimate[n - num_targets * dim:] = np.tile(
                x0_mean_prior_estimate[:dim], num_targets)
        else:
            raise NotImplementedError
    Ical0 = np.hstack((x0_mean_prior_estimate, u0.repeat(max(1, Delta) * use_information_vector)))  # \mathcal{I}_{0}
    Ical_expectation0 = np.hstack(
        (x0_mean_prior_estimate, u0.repeat(max(1, Delta) * use_information_vector)))  # E[\mathcal{I}_{0}]
    Ical_apriori0 = Ical0.copy()
    ecal_expectation0 = np.hstack((x0_mean, u0.repeat(max(1, Delta) * use_information_vector))) - Ical0  # E[[e, 0, ..., 0]] with e = x - x_hat
    Sigma_Ical0 = Ical0.reshape(-1, 1).dot(Ical0.reshape(1, -1))  # WARNING: if x0_mean_prior_estimate is non-deterministic: use expectation E[Ical0.reshape(-1, 1).dot(Ical0.reshape(1, -1))]
    # WARNING: (centered) covariance between (np.hstack((x0_mean, u0.repeat(max(1, Delta) * use_information_vector)))) and Ical0
    #  as well as its transposed both need to be subtracted in computation of Sigma_ecal0:
    Sigma_ecal0 = np.hstack((np.vstack((Sigma0, np.zeros(shape=(m - n, n)))), np.zeros(shape=(m, m - n)))) + \
                  Sigma_Ical0 - Ical_expectation0.reshape(-1, 1).dot(Ical_expectation0.reshape(1, -1)) + \
                  ecal_expectation0.reshape(-1, 1).dot(ecal_expectation0.reshape(1, -1))  # \Sigma_{0}^{ecal} (UNCENTERED/CENTERED covariance matrices of ecal=[e, 0, ..., 0] (mean of e=x-x_hat is zero))
    Sigma_ecal_sqrt0 = sqrtm_psd(Sigma_ecal0)  # \Sigma_{0}^{ecal}^{0.5}  [only used if "use_square_root_filter == True"]
    Sigma_Icalecal0 = Ical_expectation0.reshape(-1, 1).dot(ecal_expectation0.reshape(1, -1))

    Sigma_x0 = Sigma_ecal0 - ecal_expectation0.reshape(-1, 1).dot(ecal_expectation0.reshape(1, -1)) \
               + Sigma_Ical0 - Ical_expectation0.reshape(-1, 1).dot(Ical_expectation0.reshape(1, -1)) \
               + Sigma_Icalecal0 - Ical_expectation0.reshape(-1, 1).dot(ecal_expectation0.reshape(1, -1)) \
               + (Sigma_Icalecal0 - Ical_expectation0.reshape(-1, 1).dot(ecal_expectation0.reshape(1,
                                                                                                   -1))).transpose()  # \Sigma_{0}^{xcal} (CENTERED covariance matrices of xcal_{k}=[x_{k-\Delta}, u_{k-\Delta}, ..., u_{k-1}])

    return Ical0, Sigma_x0, Sigma_ecal0, Sigma_Icalecal0, Sigma_Ical0, Sigma_ecal_sqrt0, x0_mean_prior_estimate, Ical_expectation0, ecal_expectation0, Ical_apriori0


def lqg_solve_control_problem_jerkpenalization(N, dim, num_targets, gamma, saccade_times, system_matrices, m, l, Delta=0, Kk=None):
    """
    Solves a given linear-quadratic Gaussian control problem.
    Variant from Todorov05 (with jerk penalization and delayed observations from Todorov98_thesis),
    without linearization of state-dependent observation noise (specific choice used here actually only supports dim = 1...),
    and with *actual (non-observable) realization* of quadratic state costs in objective function.
    INFO: Should be used together with lqg_solve_observation_problem.
    :param N: number of time steps (excluding final step) [int]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param num_targets: number of targets [if system_dynamics=="E-LQG": including initial position] (can be used for via-point tasks) [int]
    :param gamma: visual position noise weight (only used for E-LQG) [float]
    :param saccade_times: array of indices that correspond to saccade times [see n_s in paper] (only used for E-LQG) [num_targets-array]
    :param system_dynamics: which dynamics to use ["LQG" or "E-LQG"]
    :param m: dimension of information vector [int]
    :param l: dimension of observation vector [int]
    :param Delta: observation time lag in time steps (experimental!; default: 0) [int]
    :param Kk: Kalman filter gain matrices [list of N m*l-arrays]
    :return:
        - L: optimal feedback gain matrices [list of N dim*m-arrays]
        - S_x: matrices of optimal cost-to-go functions that apply to true states [list of (N+1) m*m-arrays]
        - S_e: matrices of optimal cost-to-go functions that apply to estimation errors [list of (N+1) m*m-arrays]
        - alpha: constants of optimal cost-to-go functions [list of (N+1) floats]
    """

    (A, B, C, D, Acal, Bcal, Ccal_1, Ccal_2, Dcal, get_current_Hk, get_current_Gk, Gk_constant,
     Q, R, M, I_p, I_x, C_p, Bhelp, Dhelp, Dhelp_sum, Dcal_help) = system_matrices

    #assert Delta == 0, "This implementation of the LQG solution only supports Delta=0."
    assert dim == 1, "This implementation of the LQG solution only supports dim=1."

    if Kk is None:  # INFO: Kk[i] corresponds to Kcal[i].dot(V_sqrt_inv) (with V_sqrt_inv also depending on i...)
        # if no Kalman matrices are given, use zero matrices (i.e., compute optimal open-loop solution)
        Kk = []
        for i in range(N):
            Kk.append(np.zeros(shape=(m, l)))

    RI1 = R.dot(I_p[1])
    I1RI1 = I_p[1].transpose().dot(RI1)

    L = [None] * (N)
    L[N] = I_p[1]  # np.hstack((np.zeros(shape=(dim, m - dim)), np.eye(dim)))
    alpha = [None] * (N + 1)
    alpha[N] = np.trace(Q[N].dot(Dhelp_sum))
    S_x = [None] * (N + 1)
    S_x[N] = M.transpose().dot(Q[N]).dot(M)  # + Shelp
    S_e = [None] * (N + 1)
    S_e[N] = I1RI1

    for j in range(N - 1, -1, -1):  # compute down to L_i, S_i, P_i, and alpha_i (in Todorov's notation)
        # Iterate on modified Riccati Equation:
        Bcal_Siplus1 = Bcal.transpose().dot(S_x[j + 1])
        Ccal_2_Siplus1 = Ccal_2.transpose().dot(S_x[j + 1] + S_e[j + 1])
        Shelp_1 = RI1 - Bcal_Siplus1.dot(Acal) - Ccal_2_Siplus1.dot(Ccal_1)
        try:
            if dim == 1:
                L[j] = np.linalg.solve(R + Bcal_Siplus1.dot(Bcal) + Ccal_2_Siplus1.dot(Ccal_2),
                                       Shelp_1)
            else:
                L[j] = np.linalg.inv(R + Bcal_Siplus1.dot(Bcal) + Ccal_2_Siplus1.dot(Ccal_2)).dot(Shelp_1)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('ERROR in Discrete Riccati Equation: Singular matrix!\n')
            else:
                raise ValueError

        Hk, _ = get_current_Hk(j - Delta, saccade_times, dim, num_targets)  # H_{i-\Delta}
        Gk = get_current_Gk(dim, num_targets, Hk, gamma)

        helperSx = I_x.transpose().dot(
            sum([(Kk[j].dot(Gk_i)).transpose().dot(S_e[j + 1]).dot(Kk[j].dot(Gk_i)) for Gk_i in Gk])).dot(I_x)
        helperSe = Kk[j].dot(Hk).dot(I_x)
        helperalpha = np.trace(
            S_e[j + 1].dot(Kk[j].dot(Gk_constant.dot(Gk_constant.transpose())).dot(Kk[j].transpose())))

        Shelp = np.zeros(shape=(m, m))
        for k in range(1, Delta + 1):
            Shelp += np.dot(C_p[k].T, Q[j]).dot(C_p[k])

        S_x[j] = Acal.transpose().dot(S_x[j + 1]).dot(Acal) + (
                    M.transpose().dot(Q[j]).dot(M) + I1RI1 + helperSx) - Shelp_1.transpose().dot(L[j])
        S_e[j] = (Acal - helperSe).transpose().dot(S_e[j + 1]).dot(Acal - helperSe) + Shelp_1.transpose().dot(L[j])
        alpha[j] = alpha[j + 1] + np.trace((S_x[j + 1] + S_e[j + 1]).dot(Dcal).dot(Dcal.transpose())) + np.trace(
            Q[j].dot(Dhelp_sum)) + helperalpha

    return L, S_x, S_e, alpha


def lqg_solve_control_problem(N, dim, num_targets, gamma, saccade_times, system_matrices, m, l, Delta=0, Kk=None):
    """
    Solves a given linear-quadratic Gaussian control problem.
    Variant from Todorov05,
    without linearization of state-dependent observation noise (specific choice used here actually only supports dim = 1...).
    INFO: Should be used together with lqg_solve_observation_problem.
    :param N: number of time steps (excluding final step) [int]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param num_targets: number of targets [if system_dynamics=="E-LQG": including initial position] (can be used for via-point tasks) [int]
    :param gamma: visual position noise weight (only used for E-LQG) [float]
    :param saccade_times: array of indices that correspond to saccade times [see n_s in paper] (only used for E-LQG) [num_targets-array]
    :param system_dynamics: which dynamics to use ["LQG" or "E-LQG"]
    :param m: dimension of information vector [int]
    :param l: dimension of observation vector [int]
    :param Delta: observation time lag in time steps (experimental!; default: 0) [int]
    :param Kk: Kalman filter gain matrices [list of N m*l-arrays]
    :return:
        - L: optimal feedback gain matrices [list of N dim*m-arrays]
        - S_x: matrices of optimal cost-to-go functions that apply to true states [list of (N+1) m*m-arrays]
        - S_e: matrices of optimal cost-to-go functions that apply to estimation errors [list of (N+1) m*m-arrays]
        - alpha: constants of optimal cost-to-go functions [list of (N+1) floats]
    """

    (A, B, C, D, Acal, Bcal, Ccal_1, Ccal_2, Dcal, get_current_Hk, get_current_Gk, Gk_constant,
     Q, R, M, I_p, I_x, C_p, Bhelp, Dhelp, Dhelp_sum, Dcal_help) = system_matrices

    if Kk is None:  # INFO: Kk[i] corresponds to Kcal[i].dot(V_sqrt_inv) (with V_sqrt_inv also depending on i...)
        # if no Kalman matrices are given, use zero matrices (i.e., compute optimal open-loop solution)
        Kk = []
        for i in range(N):
            Kk.append(np.zeros(shape=(m, l)))

    L = [None] * (N)
    alpha = [None] * (N + 1)
    alpha[N] = np.trace(Q[N].dot(Dhelp_sum))  #WARNING: should equal zero (since Dhelp_sum should equal zero) if Delta == 0
    S_x = [None] * (N + 1)
    S_x[N] = M.transpose().dot(Q[N]).dot(M)  # + Shelp
    S_e = [None] * (N + 1)
    S_e[N] = np.zeros(shape=(m, m))

    for j in range(N - 1, -1, -1):  # compute down to L_i, S_i, P_i, and alpha_i (in Todorov's notation)
        # Iterate on modified Riccati Equation:
        Bcal_Siplus1 = Bcal.transpose().dot(S_x[j + 1])
        Ccal_2_Siplus1 = Ccal_2.transpose().dot(S_x[j + 1] + S_e[j + 1]) ##<-should equal zero
        Shelp_1 = - Bcal_Siplus1.dot(Acal) ##<-should equal zero
        try:
            if True: #dim == 1:  #TODO
                L[j] = np.linalg.solve(R + Bcal_Siplus1.dot(Bcal) + Ccal_2_Siplus1.dot(Ccal_2),
                                   Shelp_1)
            else:
                L[j] = np.linalg.inv(R + Bcal_Siplus1.dot(Bcal) + Ccal_2_Siplus1.dot(Ccal_2)).dot(Shelp_1)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('ERROR in Discrete Riccati Equation: Singular matrix!\n')
            else:
                raise ValueError

        Hk, _ = get_current_Hk(j - Delta, saccade_times, dim, num_targets)  # H_{i-\Delta}
        Gk = get_current_Gk(dim, num_targets, Hk, gamma)

        helperSx = I_x.transpose().dot(
            sum([(Kk[j].dot(Gk_i)).transpose().dot(S_e[j + 1]).dot(Kk[j].dot(Gk_i)) for Gk_i in Gk])).dot(I_x)
        helperSe = Kk[j].dot(Hk).dot(I_x)
        helperalpha = np.trace(
            S_e[j + 1].dot(Kk[j].dot(Gk_constant.dot(Gk_constant.transpose())).dot(Kk[j].transpose())))

        Shelp = np.zeros(shape=(m, m))
        for k in range(1, Delta + 1):
            Shelp += np.dot(C_p[k].T, Q[j]).dot(C_p[k])

        S_x[j] = Acal.transpose().dot(S_x[j + 1]).dot(Acal) + (
                M.transpose().dot(Q[j]).dot(M) + helperSx) - Shelp_1.transpose().dot(L[j])
        S_e[j] = (Acal - helperSe).transpose().dot(S_e[j + 1]).dot(Acal - helperSe) + Shelp_1.transpose().dot(L[j])
        alpha[j] = alpha[j + 1] + np.trace((S_x[j + 1] + S_e[j + 1]).dot(Dcal).dot(Dcal.transpose())) + np.trace(
            Q[j].dot(Dhelp_sum)) + helperalpha

    return L, S_x, S_e, alpha


def lqg_compute_optimal_costs(Ical0, Sigma_x0, S_x, S_e, alpha):
    """
    Computes optimal (expected) cost, given initial state mean and covariance.
    :param Ical0: (initial) internal state estimate mean [m-array]
    :param Sigma_x0: (initial) internal (true) state covariance [m*m-array]
    :param S_x: matrices of optimal cost-to-go functions that apply to true states [list of (N+1) m*m-arrays]
    :param S_e: matrices of optimal cost-to-go functions that apply to estimation errors [list of (N+1) m*m-arrays]
    :param alpha: constants of optimal cost-to-go functions [list of (N+1) floats]
    :return: optimal (expected) costs J [float]
    """
    J = Ical0.transpose().dot(S_x[0]).dot(Ical0) + np.trace((S_x[0] + S_e[0]).dot(Sigma_x0)) + alpha[
        0]  # using lqg_solve_control_problem()

    return J


def lqg_solve_observation_problem(N, dim, num_targets, gamma, saccade_times, system_matrices, initial_state_matrices, m, l,
                                  Delta=0, L=None,
                                  use_square_root_filter=True):
    """
    Solves a given linear-quadratic Gaussian observation problem.
    :param N: number of time steps (excluding final step) [int]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param num_targets: number of targets [if system_dynamics=="E-LQG": including initial position] (can be used for via-point tasks) [int]
    :param gamma: visual position noise weight (only used for E-LQG) [float]
    :param saccade_times: array of indices that correspond to saccade times [see n_s in paper] (only used for E-LQG) [num_targets-array]
    :param system_matrices: tuple of relevant system matrices (see definition below) that only need to be computed once at the beginning [tuple]
    :param initial_state_matrices: tuple of initial state vectors/matrices (see definition below) that only need to be computed once at the beginning [tuple]
    :param m: dimension of information vector [int]
    :param l: dimension of observation vector [int]
    :param Delta: observation time lag in time steps (experimental!; default: 0) [int]
    :param L: optimal feedback gain matrices [list of N dim*m-arrays]
    :param use_square_root_filter: whether to use the square root filter to update Kalman matrices (default: True) [bool]
    :return:
        - Kk: Kalman filter gain matrices [list of N m*l-arrays]
        - Kcal: matrices used for Kalman filter gains and covariance updates (only incorprates square-root of inverse of innovation covariance V) [list of N m*l-arrays]
        - Sigma_Ical: covariance matrices of internal state estimates [list of (N + Delta + 1) m*m-arrays]
        - Sigma_ecal: covariance matrices of estimation error ecal (ecal = true state (~x) - estimated state mean (~Ical)) [list of (N + Delta + 1) m*m-arrays]
        - Sigma_Icalecal: covariance matrices between internal state estimate Ical and estimation error ecal [list of (N + Delta + 1) m*m-arrays]
    """

    (A, B, C, D, Acal, Bcal, Ccal_1, Ccal_2, Dcal, get_current_Hk, get_current_Gk, Gk_constant,
     Q, R, M, I_p, I_x, C_p, Bhelp, Dhelp, Dhelp_sum, Dcal_help) = system_matrices

    if L is None:
        # if no control matrices are given, use zero matrices (i.e., compute optimal observation gains for uncontrolled dynamics)
        L = []
        for i in range(N):
            L.append(np.zeros(shape=(dim, m)))

    Sigma_ecal = [None] * (N + Delta + 1)
    Sigma_Ical = [None] * (N + Delta + 1)
    Sigma_Icalecal = [None] * (N + Delta + 1)
    Sigma_ecal_sqrt = [None] * (N + 1)

    _, _, Sigma_ecal[0], Sigma_Icalecal[0], Sigma_Ical[0], Sigma_ecal_sqrt[0] = initial_state_matrices[:6]

    Kcal = [None] * N
    V = [None] * N
    Kk = [None] * N

    for i in range(0, N):
        # 1. Get observation matrices
        Hk, f_k = get_current_Hk(i - Delta, saccade_times, dim, num_targets)  # H_{i-\Delta}
        # Fk = np.diag(np.hstack((np.array([sigma_H ** 2, sigma_Hdot ** 2, sigma_e ** 2]),
        #                         gamma * np.linalg.norm(Hk_times_x_hat_expectation[dim * 3:].reshape(-1, dim),
        #                                                axis=1) ** 2)).repeat(dim))
        Gk = get_current_Gk(dim, num_targets,
                            Hk, gamma)  # related to G_{k-\Delta}(X_{k-\Delta}) from Todorov98_thesis, p.40 (but using sum formulation from Todorov05; note that Gk thus is a list of matrices, and the state-independent parts of every matrix are stored in Gk_constant)


        # 2. Compute Kalman Filter Matrix
        if use_square_root_filter:  ## SQUARE ROOT FILTER (Todorov1998, p.43)
            """
            INFO: In general, K_tilde and V_sqrt_inv might differ between both variants from Todorov 
            (independent of whether the square root filter is used), 
            but K_tilde.dot(V_sqrt_inv) (and thus also Kcal[i].dot(v[i])) should match!
            Sigma should also match except for rounding errors (in this respect, square root filter should be more stable).
            """
            M_sqrt_helper_matrix = sqrtm_psd(Dcal.dot(Dcal.transpose()) + \
                                             (Ccal_1 + Ccal_2.dot(L[i])).dot(Sigma_Ical[i]).dot(
                                                 (Ccal_1 + Ccal_2.dot(L[
                                                                          i])).transpose()))  # the last term should correspond to C.dot(u[i - Delta].dot(u[i - Delta].transpose())).dot(C.transpose())
            assert not np.isnan(M_sqrt_helper_matrix).any()
            helper_matrix = np.vstack((np.hstack(
                (Hk.dot(I_x).dot((Sigma_ecal_sqrt[i])), sqrtm_psd(Gk_constant.dot(Gk_constant.transpose()) +
                                                                  sum([Gk_idx.dot(I_x.dot(Sigma_Ical[i] +
                                                                                          Sigma_ecal[i] +
                                                                                          Sigma_Icalecal[i] +
                                                                                          Sigma_Icalecal[
                                                                                              i].transpose()).dot(
                                                                      I_x.transpose())).dot(Gk_idx.transpose()) for
                                                                       Gk_idx in Gk])),
                 np.zeros(shape=(l, m)))), np.hstack(
                (Acal.dot((Sigma_ecal_sqrt[i])), np.zeros(shape=(m, l)), M_sqrt_helper_matrix))))
            helper_matrix_Q, helper_matrix_R = np.linalg.qr(helper_matrix.transpose(), mode="complete")

            # helper_matrix_R[np.abs(helper_matrix_R) < 10e-16] = 0
            try:
                V_sqrt_inv = np.linalg.inv(helper_matrix_R.transpose()[:l, :l])
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    # print(f"WARNING: Cannot invert V[{max(0, i - Delta)}] as it is singular. Using pseudo-inverse to proceed.")
                    V_sqrt_inv = np.linalg.pinv(helper_matrix_R.transpose()[:l, :l])
                else:
                    raise np.linalg.LinAlgError
            else:
                if np.isnan(V_sqrt_inv).any() or np.max(np.abs(V_sqrt_inv)) > 10e+30:
                    V_sqrt_inv = np.linalg.pinv(helper_matrix_R.transpose()[:l, :l])
                    if np.isnan(V_sqrt_inv).any():
                        raise ValueError("ERROR - NaN in Kalman Filter Update!")

            K_tilde = helper_matrix_R.transpose()[l:, :l]
            Sigma_ecal_sqrt[i + 1] = helper_matrix_R.transpose()[l:, l:(l + m)]
        else:  # WARNING: ASSUMES that V[i - Delta] is symmetric positive definite (e.g., this does not work for Gk_constant == 0, Gk == 0!)
            # V[i - Delta] = Hk.dot(I_x).dot(Sigma_ecal[i]).dot(Hk.dot(I_x).transpose()) + Fk  # V_{i-\Delta}
            V[i - Delta] = Hk.dot(I_x).dot(Sigma_ecal[i]).dot(Hk.dot(I_x).transpose()) + Gk_constant.dot(
                Gk_constant.transpose()) + \
                           sum([Gk_i.dot(I_x.dot(
                               Sigma_ecal[i] + Sigma_Ical[i] + Sigma_Icalecal[i] + Sigma_Icalecal[i].transpose()).dot(
                               I_x.transpose())).dot(
                               Gk_i.transpose()) for Gk_i in Gk])  # V_{i-\Delta}
            # V[i - Delta][np.abs(V[i - Delta]) < 10e-16] = 0
            try:
                V_sqrt_inv = np.linalg.inv(sqrtm_psd(V[i - Delta]))
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    # print(f"WARNING: Cannot invert V[{max(0, i - Delta)}] as it is singular. Using pseudo-inverse to proceed.")
                    V_sqrt_inv = np.linalg.pinv(sqrtm_psd(V[i - Delta]))
                else:
                    raise np.linalg.LinAlgError
            K_tilde = Acal.dot(Sigma_ecal[i]).dot(Hk.dot(I_x).transpose()).dot(
                V_sqrt_inv.transpose())  # \tilde{K}_{i-\Delta}
        Kcal[i] = K_tilde  # np.vstack((K_tilde, np.zeros(shape=(m - n, l))))  #\mathcal{K}_{i-\Delta}
        Kk[i] = Kcal[i].dot(V_sqrt_inv)
        # INFO: Kalman Filter Matrix K_n from paper corresponds to Kk[i] for i==n

        # 3. Update internal state estimates Sigma_Ical, Sigma_ecal, and Sigma_Icalecal [necessary for Kalman filter updates, and to compute total (unconditional) variance]
        if use_square_root_filter:
            Sigma_ecal[i + 1] = Sigma_ecal_sqrt[i + 1].dot(Sigma_ecal_sqrt[i + 1].transpose())
        else:
            Sigma_ecal[i + 1] = Acal.dot(Sigma_ecal[i]).dot(Acal.transpose()) - K_tilde.dot(K_tilde.transpose()) + \
                                Dcal.dot(Dcal.transpose()) + \
                                (Ccal_1 + Ccal_2.dot(L[i])).dot(Sigma_Ical[i]).dot(
                                    (Ccal_1 + Ccal_2.dot(L[i])).transpose())
            # the first row of the right side of the above equation is equivalent to "Acal.dot(Sigma_ecal[i]).dot((Acal - Kk[i].dot(Hk).dot(I_x)).transpose())"
        A_plus_B_times_Li = Acal + Bcal.dot(L[i])
        Sigma_Ical[i + 1] = A_plus_B_times_Li.dot(Sigma_Ical[i]).dot(A_plus_B_times_Li.transpose()) + \
                            A_plus_B_times_Li.dot(Sigma_Icalecal[i]).dot(
                                (Kk[i].dot(Hk).dot(I_x)).transpose()) + \
                            (Kk[i].dot(Hk).dot(I_x)).dot(Sigma_Icalecal[i].transpose()).dot(
                                A_plus_B_times_Li.transpose()) + \
                            K_tilde.dot(
                                K_tilde.transpose())  # this row is equivalent to "Kk[i].dot(Hk).dot(I_x).dot(Sigma_ecal[i]).dot(Acal.transpose())"
        Sigma_Icalecal[i + 1] = A_plus_B_times_Li.dot(Sigma_Icalecal[i]).dot(
            (Acal - (Kk[i].dot(Hk).dot(I_x))).transpose())

    # 4. Predict remaining internal state mean and covariance estimates / stage costs (only in case of observation time lags):
    Sigma_ecal_helper = Sigma_ecal[N]
    Sigma_Ical_helper = Sigma_Ical[N]
    Sigma_Icalecal_helper = Sigma_Icalecal[N]
    D_helper = np.zeros(shape=(m, m))
    C_helper = np.zeros(shape=(m, m))
    for ii in range(1, Delta + 1):
        Sigma_ecal_helper = Acal.dot(Sigma_ecal_helper).dot(Acal.transpose())
        Sigma_Ical_helper = (Acal + Bcal.dot(I_p[1])).dot(Sigma_Ical_helper).dot((Acal + Bcal.dot(I_p[1])).transpose())
        Sigma_Icalecal_helper = (Acal + Bcal.dot(I_p[1])).dot(Sigma_Icalecal_helper).dot(Acal.transpose())
        D_helper += Dcal_help[ii]
        C_helper = Acal.dot(C_helper).dot(Acal.transpose()) + (Ccal_1 + Ccal_2.dot(I_p[1])).dot(
            Sigma_Ical[N]).dot((Ccal_1 + Ccal_2.dot(I_p[1])).transpose())  # INFO: assumes that u_{N-1} is constantly applied afterwards (note that u_{k-\Delta} dynamics are applied via Acal term)
        Sigma_ecal[N + ii] = Sigma_ecal_helper + D_helper + C_helper
        Sigma_Ical[N + ii] = Sigma_Ical_helper
        Sigma_Icalecal[N + ii] = Sigma_Icalecal_helper

    return Kk, Kcal, Sigma_Ical, Sigma_ecal, Sigma_Icalecal


def lqg_forward_simulation(L, Kk, Kcal, Sigma_Ical, Sigma_ecal, Sigma_Icalecal,
                           N, dim, num_targets, x0_mean, u0, Sigma0,
                           system_matrices, initial_state_matrices, n, m, l,
                           gamma, saccade_times, Delta=0,
                           minimum_computations=False,
                           use_information_vector=False):
    """
    Running forward loop using optimal feedback and Kalman filter gains.
    :param L: optimal feedback gain matrices [list of N dim*m-arrays]
    :param Kk: Kalman filter gain matrices [list of N m*l-arrays]
    :param Kcal: matrices used for Kalman filter gains and covariance updates (only incorprates square-root of inverse of innovation covariance V) [list of N m*l-arrays]
    :param Sigma_Ical: covariance matrices of internal state estimates [list of (N + Delta + 1) m*m-arrays]
    :param Sigma_ecal: covariance matrices of estimation error ecal (ecal = true state (~x) - estimated state mean (~Ical)) [list of (N + Delta + 1) m*m-arrays]
    :param Sigma_Icalecal: covariance matrices between internal state estimate Ical and estimation error ecal [list of (N + Delta + 1) m*m-arrays]
    :param N: number of time steps (excluding final step) [int]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param num_targets: number of targets [if system_dynamics=="E-LQG": including initial position] (can be used for via-point tasks) [int]
    :param x0_mean: initial state expectation ((flattened) position, velocity, muscle force, muscle excitation, [initial position (only if system_dynamics=="E-LQG"),] target position(s)) [n-array]
    :param u0: initial control value (only used if "use_information_vector == True") [dim-array]
    :param Sigma0: initial state covariance matrix [n*n-array]
    :param system_matrices: tuple of relevant system matrices (see definition below) that only need to be computed once at the beginning [tuple]
    :param initial_state_matrices: tuple of initial state vectors/matrices (see definition below) that only need to be computed once at the beginning [tuple]
    :param n: dimension of state vector [int]
    :param m: dimension of information vector [int]
    :param l: dimension of observation vector [int]
    :param gamma: visual position noise weight (only used for E-LQG) [float]
    :param saccade_times: array of indices that correspond to saccade times [see n_s in paper] (only used for E-LQG) [num_targets-array]
    :param Delta: observation time lag in time steps (experimental!; default: 0) [int]
    :param minimum_computations: if True, realized costs and other stuff are not computed and printed (useful in optimizations etc.) [bool]
    :param use_information_vector: whether to augment state vectors with latest controls (needs to be True if Delta > 0) [bool]
    :return:
        - J_cp: optimal (realized) costs (only if "minimum_computations==True")
          [slightly different internal computation than "J_sample"] [float]
        - J_sample: optimal (realized) costs (only if "minimum_computations==True") [float]
        - x_expectation: mean of expected optimal state sequence (except for target components,
          which correspond to internal estimates at same time) [(N+1)*m-array]
        - u_expectation: expected optimal control sequence [N*dim-array]
        - Sigma_x: variance of expected optimal (true) state sequence [list of (N+Delta+1) m*m-arrays]
        - Ical: sequence internal state estimate means [list of (N+1) m-arrays]
        - x: true state sequence [(N+1)*n-array]
        - u: optimal control sequence (depending on realization of observations y) [N*dim-array]
        - y: observation sequence [(N+1)*l-array]
    """

    (A, B, C, D, Acal, Bcal, Ccal_1, Ccal_2, Dcal, get_current_Hk, get_current_Gk, Gk_constant,
     Q, R, M, I_p, I_x, C_p, Bhelp, Dhelp, Dhelp_sum, Dcal_help) = system_matrices

    # 1. Setup and initialize internal and external variables
    # 1.1. Setup variables
    Ical = [None] * (N + 1)
    Sigma_x = [None] * (N + Delta + 1)

    ecal = [None] * (N + 1)
    Sigma_ecal_sqrt = [None] * (N + 1)  # [only used if "use_square_root_filter == True"]

    Ical_expectation = [None] * (N + Delta + 1)
    ecal_expectation = [None] * (N + Delta + 1)

    Ical_apriori = [None] * (N + 1)
    Sigma_apriori = [None] * (N + 1)
    Sigma_conditional = [None] * (N + 1)


    x = np.zeros((N + 1, n))  # true state vectors x_{i}, i=0..N
    u = np.zeros((N, dim))
    y = np.zeros((N + 1, l))
    x_hat = np.zeros((N + 1, n))  # state mean estimates \hat{x_{i-\Delta|i-1}}, i=0..N

    #x_expectation = np.zeros((N + 1, n))  # expectation of true state vectors E[x_{i}], i=0..N
    u_expectation = np.zeros((N, dim))
    x_hat_apriori = np.zeros((N + 1, n))  # Ical_apriori/x_hat_apriori corresponds to sequence of state estimates with optimal closed-loop control (given some Kalman matrices, e.g., via iteration between Kalman and Control Law Matrix updates), but open-loop forward pass (Kalman matrices are not used in system equations)

    Sigma_u = [None] * (N)

    eta = [None] * (N + 1)
    delta = [None] * (N + 1)
    omega = [None] * (N + 1)

    y_err = [None] * (N + 1)

    # 1.2. Initialize variables
    (Ical[0], Sigma_x[0], Sigma_ecal[0], Sigma_Icalecal[0], Sigma_Ical[0], Sigma_ecal_sqrt[0], \
    x0_mean_prior_estimate, Ical_expectation[0], ecal_expectation[0], Ical_apriori[0]) = initial_state_matrices

    x_hat[0] = I_x.dot(Ical[0])
    x_hat_apriori[0] = I_x.dot(Ical_apriori[0])
    Sigma_apriori[0] = I_x.dot(Sigma_ecal[0]).dot(I_x.transpose())  # Sigma0 (or Sigma0_minusDelta)
    Sigma_conditional[0] = I_x.dot(Sigma_ecal[0]).dot(I_x.transpose())  # Sigma0 (or Sigma0_minusDelta)

    x0 = np.random.multivariate_normal(x0_mean, Sigma0)  # sample initial state
    x[-Delta] = x0
    ecal[0] = np.hstack((x0 - x0_mean_prior_estimate, np.zeros((dim,)).repeat(max(1, Delta) * use_information_vector)))  # [e, 0, ..., 0] with e = x - x_hat

    # 1.3. Compute previous (but still unobserved) data in case of observation time lags
    x_expectation_helper = x0_mean
    for i in range(-Delta, 0):
        u[i] = u0

        eta[i] = np.random.normal()
        delta[i] = np.random.multivariate_normal(mean=np.zeros(n, ), cov=np.eye(n))
        x[i + 1] = A.dot(x[i]) + (B + eta[i] * C).dot(u[i]) + D.dot(delta[i])

        x_expectation_helper = A.dot(x_expectation_helper) + B.dot(u[i])


    # 2. Run forward loop
    J_cp = 0 if not minimum_computations else None
    J_sample = 0 if not minimum_computations else None
    #J_sample_state = 0 if not minimum_computations else None
    for i in range(0, N):

        # 2.1. Observation dynamics
        Hk, f_k = get_current_Hk(i - Delta, saccade_times, dim, num_targets)  # H_{i-\Delta}
        Hk_times_x_hat = Hk.dot(x_hat[i])  # H_{i-\Delta} * \hat{X}_{i-\Delta | i-1}
        Gk = get_current_Gk(dim, num_targets,
                            Hk, gamma)  # related to G_{k-\Delta}(X_{k-\Delta}) from Todorov98_thesis, p.40 (but using sum formulation from Todorov05; note that Gk thus is a list of matrices, and the state-independent parts of every matrix are stored in Gk_constant)
        omega[i] = np.random.multivariate_normal(mean=np.zeros(l, ), cov=np.eye(l))
        y[i] = Hk.dot(x[i - Delta]) + Gk_constant.dot(omega[i]) + sum(
            [omega[i] * np.abs(Gk_idx.dot(x[i - Delta])) for idx, Gk_idx in
             enumerate(Gk)])  # WARNING: if dim > 1 and system_dynamics == "E-LQG", Fk should be used instead of Gk_constant and Gk!
        # Hk_times_x_hat_expectation = Hk.dot(I_x.dot(Ical_expectation[
        #                                                 i]))  # H_{i-\Delta} * \hat{X}_{i-\Delta | i-1}  #WARNING: not consistent with definition from Todorov1998_thesis, as this depends on realization of u (i.e., of y)
        # Fk = np.diag(np.hstack((np.array([sigma_H ** 2, sigma_Hdot ** 2, sigma_e ** 2]),
        #                         gamma * np.linalg.norm(Hk_times_x_hat_expectation[dim * 3:].reshape(-1, dim),
        #                                                axis=1) ** 2)).repeat(dim))
        # omega[i] = np.random.multivariate_normal(mean=np.zeros(l, ), cov=Fk)
        # y[i] = Hk.dot(x[i - Delta]) + omega[i]

        # 2.2. Control dynamics
        u[i] = L[i].dot(Ical[i])  # u_{i}
        u_expectation[i] = L[i].dot(Ical_expectation[i])
        Sigma_u[i] = L[i].dot(Sigma_Ical[i]).dot(L[i].transpose()) - u_expectation[i].dot(u_expectation[i].transpose())
        A_plus_B_times_Li = Acal + Bcal.dot(L[i])


        # 2.3. Update computation of true costs (which depends on realization of random variables)
        if not minimum_computations:  # INFO: both J and J_cp depend on realization of u (i.e., of y), and use conditional variance estimates!
            # Optimal cost computation update
            D_helper = np.zeros(shape=(n, n))
            M_helper_1 = np.eye(n).dot(I_x)
            M_helper_2 = np.zeros(shape=(n, m))
            for ii in range(1, Delta + 1):
                D_helper += Dhelp[
                    ii]  # Dhelp[ii] == I_x.dot(np.dot((np.linalg.matrix_power(Acal, (ii - 1))), Dcal)).dot(I_x.dot(np.dot((np.linalg.matrix_power(Acal, (ii - 1))), Dcal)).transpose())
                M_helper_1 = A.dot(M_helper_1)
                M_helper_2 += Bhelp[ii]
            x_hat_forward_prediction = (M_helper_1 + M_helper_2).dot(Ical[i] + ecal[
                i])  # x_hat_forward_prediction == I_x.dot(np.linalg.matrix_power(Acal, Delta)).dot(Ical[i] + ecal[i]) [with (Ical[i] + ecal[i]) corresponding to xcal[i]...]
            # WARNING: J_cp depends on trajectory realization, since x_hat_forward_prediction depends on  xcal[i] = Ical[i] + ecal[i] and u[i] depends on y[i] (similar to J_sample, in contrast to total expected cost J).
            Sigma_forward_prediction = D_helper  # Sigma_helper + D_helper + C_helper
            J_cp += x_hat_forward_prediction.transpose().dot(Q[i]).dot(x_hat_forward_prediction) + np.trace(
                Q[i].dot(Sigma_forward_prediction))
            J_sample += x[i].transpose().dot(Q[i]).dot(x[i])
            #J_sample_state += x[i].transpose().dot(Q[i]).dot(x[i])
            J_cp += (u[i]).transpose().dot(R).dot(u[i])
            J_sample += (u[i]).transpose().dot(R).dot(u[i])

        # 2.3. Update state (estimate) variables
        # 2.3.1 Update internal state estimate Ical [necessary for forward loop; Sigma_Ical, Sigma_ecal, and Sigma_Icalecal were already computed during Kalman Filter computation]
        y_err[i] = y[i] - Hk_times_x_hat  # related to v_{i} (but V_sqrt_inv is already included in Kk[i])
        Ical[i + 1] = Acal.dot(Ical[i]) + Bcal.dot(u[i]) + Kk[i].dot(y_err[i])  # \mathcal{I}_{i+1}
        x_hat[i + 1] = I_x.dot(Ical[i + 1])

        # 2.3.2. Update state statistics [not necessary for forward loop, but provides unique (average) movement distributions]
        # (Unconditional) state mean (independent of realization of observations y, which affect internal state estimates Ical and thus controls u)
        Ical_expectation[i + 1] = A_plus_B_times_Li.dot(Ical_expectation[i]) + Kk[i].dot(
            Hk.dot(I_x).dot(ecal_expectation[i]))
        ecal_expectation[i + 1] = (Acal - Kk[i].dot(Hk).dot(I_x)).dot(ecal_expectation[i])
        # (Unconditional) state variance
        Sigma_x[i + 1] = Sigma_ecal[i + 1] - ecal_expectation[i + 1].reshape(-1, 1).dot(
            ecal_expectation[i + 1].reshape(1, -1)) \
                         + Sigma_Ical[i + 1] - Ical_expectation[i + 1].reshape(-1, 1).dot(
            Ical_expectation[i + 1].reshape(1, -1)) \
                         + Sigma_Icalecal[i + 1] - Ical_expectation[i + 1].reshape(-1, 1).dot(
            ecal_expectation[i + 1].reshape(1, -1)) \
                         + (Sigma_Icalecal[i + 1] - Ical_expectation[i + 1].reshape(-1, 1).dot(
            ecal_expectation[i + 1].reshape(1, -1))).transpose()

        # 2.3.3. Update other state quantities (expectations of open-loop forward pass (with optimal closed-loop controls), conditional variance) [not necessary for forward loop]
        Ical_apriori[i + 1] = Acal.dot(Ical_apriori[i]) + Bcal.dot(u[i])
        x_hat_apriori[i + 1] = I_x.dot(Ical_apriori[i + 1])
        Sigma_apriori[i + 1] = A.dot(Sigma_apriori[i]).dot(A.transpose()) + D.dot(D.transpose()) + \
                               C.dot(u[i - Delta].reshape(-1, 1).dot(u[i - Delta].reshape(1, -1))).dot(
                                   C.transpose())
        Sigma_conditional[i + 1] = A.dot(Sigma_conditional[i]).dot(A.transpose()) + D.dot(
            D.transpose()) + \
                                   C.dot(u[i - Delta].reshape(-1, 1).dot(
                                       u[i - Delta].reshape(1, -1))).dot(C.transpose()) - I_x.dot(
            Kcal[i].dot(Kcal[i].transpose())).dot(I_x.transpose())

        # 2.4. True (but not (perfectly) observable) state dynamics
        eta[i] = np.random.normal()
        delta[i] = np.random.multivariate_normal(mean=np.zeros(n, ), cov=np.eye(n))
        x[i + 1] = A.dot(x[i]) + (B + eta[i] * C).dot(u[i]) + D.dot(delta[i])

        ecal[i + 1] = (Acal - Kk[i].dot(Hk).dot(I_x)).dot(ecal[i]) + eta[i - Delta] * (
                Ccal_1 + Ccal_2.dot(L[i])).dot(Ical[i]) + Dcal.dot(delta[i - Delta]) - Kk[i].dot(
            Gk_constant.dot(omega[i]) + sum([omega[i] * np.abs(Gk_idx.dot(x[i - Delta])) for idx, Gk_idx in enumerate(
                Gk)]))  # WARNING: last term might differ if Fk depends on x rather than x_hat (which should be the case, but is avoided in Todorov1998_thesis due to Fk being a simplified linear approximation, which is required for dim > 1...)

    # 3. Predict remaining internal state mean and covariance estimates / stage costs (only in case of observation time lags):
    for ii in range(1, Delta + 1):
        Ical_expectation[N + ii] = Acal.dot(Ical_expectation[N + ii - 1]) + Bcal.dot(I_p[1]).dot(Ical_expectation[
                                                                                                     N])  # INFO: assumes that u_{N-1} is constantly applied afterwards (note that u_{k-\Delta} dynamics are applied via Acal term)
        ecal_expectation[N + ii] = Acal.dot(ecal_expectation[N + ii - 1])
        Sigma_x[N + ii] = Sigma_ecal[N + ii] - ecal_expectation[N + ii].reshape(-1, 1).dot(
            ecal_expectation[N + ii].reshape(1, -1)) \
                          + Sigma_Ical[N + ii] - Ical_expectation[N + ii].reshape(-1, 1).dot(
            Ical_expectation[N + ii].reshape(1, -1)) \
                          + Sigma_Icalecal[N + ii] - Ical_expectation[N + ii].reshape(-1, 1).dot(
            ecal_expectation[N + ii].reshape(1, -1)) \
                          + (Sigma_Icalecal[N + ii] - Ical_expectation[N + ii].reshape(-1, 1).dot(
            ecal_expectation[N + ii].reshape(1, -1))).transpose()
    if not minimum_computations:
        M_helper_1 = np.eye(n).dot(I_x)
        M_helper_2 = np.zeros(shape=(n, m))
        x_hat_forward_prediction = (M_helper_1 + M_helper_2).dot(
            Ical[N] + ecal[N])  # depends on M_helper_1, M_helper_2 computed above
        Sigma_forward_prediction = sum([I_x.dot(np.dot((np.linalg.matrix_power(Acal, (ii - 1))), Dcal)).dot(
            I_x.dot(np.dot((np.linalg.matrix_power(Acal, (ii - 1))), Dcal)).transpose()) for ii in range(1, Delta + 1)])
        J_cp += x_hat_forward_prediction.transpose().dot(Q[N]).dot(x_hat_forward_prediction) + np.trace(
            Q[N].dot(Sigma_forward_prediction))
        J_sample += x[N].transpose().dot(Q[N]).dot(x[N])
        #J_sample_state += x[N].transpose().dot(Q[N]).dot(x[N])

    # 4. Final polishing, align time series of true states with internal target estimates
    Ical = np.squeeze(Ical)
    ecal = np.squeeze(ecal)
    Ical_expectation = np.squeeze(Ical_expectation)

    # Compute x_expectation, which corresponds to sequence of expected *true* states (which become available with observation time lag Delta
    # and are thus included in the internal state estimate variables only with delay, i.e., Ical_expectation[Delta:, :]),
    # except for target components, which correspond to internal estimates at same time (stored in current variables Ical_expectation[:-Delta, :] (due to observation time lag Delta))
    x_expectation = Ical_expectation[Delta:, :n]
    if Delta > 0:
        x_expectation[:, n - num_targets * dim:n] = Ical_expectation[:-Delta, n - num_targets * dim:n]
        Ical_expectation = Ical_expectation[Delta:, :]
        Sigma_x = Sigma_x[Delta:]
    else:
        x_expectation[:, n - num_targets * dim:n] = Ical_expectation[:, n - num_targets * dim:n]

    return J_cp, J_sample, x_expectation, Sigma_x, u_expectation, Ical, x, u, y


def paramfitting(user, distance, width, direction,
                 params_to_optimize,
                 param_dict_fixed,
                 control_method="LQG",
                 system_dynamics="LQG",  # only used for control_method=="LQG"
                 loss_type="MWD",
                 fitting_method="expectation-based",
                 n_simulation_samples=200,
                 manual_reactiontime_cutoff=True,
                 shift_target_to_origin=False, analysis_dim=None,
                 include_proprioceptive_target_signals=False,
                 include_proprioceptive_endeffector_signals=False,
                 modify_init_target_estimate=True,  # only used for system_dynamics=="E-LQG"
                 use_square_root_filter=True,
                 min_algorithm_iterations=1, algorithm_iterations=20, J_eps=1e-3,
                 dir_path="PointingDynamicsDataset", DATAtrials=None, trial_info_path="PD-Dataset.pickle"):
    """
    Computes parameter values such that resulting optimal (expected) trajectory of given control method
    matches reference trajectory from Pointing Dynamics Dataset (given a user, target distance and width, and movement
    direction) as close as possible (with respect to given loss function).
    :param user: user ID [1-12]
    :param distance: distance to target in px [765 or 1275]
    :param width: corresponding target width in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}]
    :param direction: movement direction ["left" or "right"]
    :param params_to_optimize: list of parameters that should be optimized [list of strings]
    :param param_dict_fixed: dictionary with name-value pairs of remaining (fixed) parameters; missing parameters are set to their default value [dict]
    :param control_method: control method to use ["2OL-Eq", "MinJerk", "LQR", or "LQG"]
    :param system_dynamics: which dynamics to use (only used if control_method=="LQG") ["LQG" or "E-LQG"]
    :param loss_type: loss function to use ["SSE", "Maximum Error", "MAE", "MKL", or "MWD" (availability also depends on "control_method")]
    :param fitting_method: whether to compare simulation and user data based on expectation or sample statistics ["expectation-based" or "sample-based"]
    :param n_simulation_samples: number of simulation samples used to compute distribution statistics (only if fitting_method=="sample-based") [int]
    :param manual_reactiontime_cutoff: whether reaction times should be removed from reference user trajectory [bool]
    :param shift_target_to_origin: whether to shift coordinate system such that target center is in origin [bool]
    :param analysis_dim: how many dimensions of user data trajectories should be considered in loss function (e.g., 2 means position and velocity only) [1-3]
    :param include_proprioceptive_target_signals: whether target position(s) can be observed in absolute coordinates
    (only usable for LQG; default: False) [bool]
    :param include_proprioceptive_endeffector_signals: whether end-effector can be observed position in absolute coordinates
    (only usable for E-LQG; default: False) [bool]
    :param modify_init_target_estimate: whether to place an incorrect initial target estimate
    (basically, the target estimate is set to initial movement position, which makes particular sense for via-point tasks)
    (only works for E-LQG or if "include_proprioceptive_target_signals==True") [bool]
    :param use_square_root_filter: whether to use the square root filter to update Kalman matrices (default: True) [bool]
    :param min_algorithm_iterations: minimum number of iterations (see "algorithm_iterations")
    :param algorithm_iterations: (maximum) number of iterations, where the optimal control and
    the optimal observation problem is solved alternately (if "J_eps" is set, early termination is possible) [int]
    :param J_eps: if relative improvement of cost function falls below "J_eps" and "min_algorithm_iterations" is reached,
    iterative solving algorithm terminates [float]
    :param dir_path: local path to Mueller's PointingDynamicsDataset (only used if "dir_path.txt" does not exist) [str]
    :param DATAtrials: trial info object (if not provided, this is loaded from trial_info_path) [pandas.DataFrame]
    :param trial_info_path: path to trial info object (if file does not exist, DATAtrials object is re-computed and stored to file) [str]
    :return: optimization result object [scipy.optimize.OptimizeResult]
    """

    if control_method != "LQG" and system_dynamics == "E-LQG":
        system_dynamics = "LQG"  # ensure consistency (system_dynamics is not directly used in this case)
    if analysis_dim is None:
        analysis_dim = 2 if control_method == "LQG" else 1

    # 1. Compute task information and reference data trajectories/distributions
    x_loc_data, x_scale_data, Nmax, dt, dim, \
    initialvalues_alltrials, T_alltrials, widthm_alltrials = pointingdynamics_data(user, distance, width, direction,
                                                              manual_reactiontime_cutoff=manual_reactiontime_cutoff,
                                                              shift_target_to_origin=shift_target_to_origin,
                                                              analysis_dim=analysis_dim,
                                                              dir_path=dir_path, DATAtrials=DATAtrials, trial_info_path=trial_info_path)
    num_targets = 1 + (system_dynamics == "E-LQG")  # number of via-point targets (default pointing task: 1 (2 if system_dynamics == "E-LQG", as initial position needs to be included then))


    # 2. Compute initial state mean "x0_mean" and covariance "Sigma0" (if not given)
    T_alltrials = [T[(system_dynamics == "LQG"):] for T in T_alltrials]  # remove initial position from target T (and thus from initial state x0; only for LQG!)
    if control_method == "MinJerk":  #incorporate acceleration in initial state and target position(s) in initial state (might be removed again later)
        x0_alltrials = [np.array(list(initialuservalues_trial) + T_trial)
                        for (T_trial, initialuservalues_trial) in zip(T_alltrials, initialvalues_alltrials)]
    else:  #incorporate zero muscle activation and excitation and target position(s) in initial state (might be removed again later)
        x0_alltrials = [np.array([initialuservalues_trial[0], initialuservalues_trial[1]] + [0] * (2 * dim) + T_trial)
                        for (T_trial, initialuservalues_trial) in zip(T_alltrials, initialvalues_alltrials)]
    x0_mean = np.mean(x0_alltrials, axis=0)
    u0 = np.zeros(shape=(dim, ))
    # Define initial covariance matrix (the variable Sigma0 corresponds to Sigma_1 in [Todorov1998])
    Sigma0 = np.cov(x0_alltrials, rowvar=False)
    Sigma0[np.abs(Sigma0) < 10e-10] = 0

    # 3. Prepare parameter (boundary) structures (params_info tuples consist of (lower bound, upper bound, [number of param entities,] type))
    if control_method == "2OL-Eq":
        params_info = {'k': (0, 500, 'continuous'), 'd': (0, 500, 'continuous')}
        params_all = ['k', 'd']
    elif control_method == "MinJerk":
        params_info = {'passage_times': (0, Nmax, 1, 'continuous_sequence')}
        params_all = ['passage_times']
    elif control_method == "LQR":
        params_info = {'r': (-3, 20, 'continuous'),
                     'velocitycosts_weight': (0, 0.1, 'continuous'),
                     'forcecosts_weight': (0, 0.001, 'continuous'),
                     'mass': (0.01, 10, 'continuous'),
                     't_const_1': (0.01, 0.5, 'continuous'),
                     't_const_2': (0.01, 0.5, 'continuous')}
        params_all = ['r', 'velocitycosts_weight', 'forcecosts_weight', 'mass', 't_const_1', 't_const_2']
    elif control_method == "LQG":
        params_info = {'r': (5, 40, 'continuous'),
                       'velocitycosts_weight': (0, 10, 'continuous'),
                       'forcecosts_weight': (0, 10, 'continuous'),
                       'mass': (0.01, 10, 'continuous'),
                       't_const_1': (0.01, 0.5, 'continuous'),
                       't_const_2': (0.01, 0.5, 'continuous'),
                       'sigma_u': (1e-09, 5, 'continuous'),
                       'sigma_c': (1e-19, 0.001, 'continuous'),
                       'sigma_s': (0, 5, 'continuous'),
                       'sigma_H': (0, 1, 'continuous'),
                       'sigma_Hdot': (0, 10, 'continuous'),
                       'sigma_frc': (0, 50, 'continuous'),
                       'sigma_e': (0, 5, 'continuous'),
                       'gamma': (4e-18, 100, 'continuous'),
                       'passage_times': (0, Nmax, 1, 'continuous_sequence'),
                       'saccade_times': (0, Nmax, 1, 'continuous_sequence'),
                       'Delta': (0, np.min((int(0.5 / dt), Nmax)), 'discrete')}  # maximum delay should not exceed 500ms nor total movement time
        if system_dynamics == "LQG":
            params_all = ['r', 'velocitycosts_weight', 'forcecosts_weight', 'mass', 't_const_1', 't_const_2',
                          'sigma_u', 'sigma_c', 'sigma_s',
                          'passage_times', 'Delta']
        elif system_dynamics == "E-LQG":
            params_all = ['r', 'velocitycosts_weight', 'forcecosts_weight', 'mass', 't_const_1', 't_const_2',
                          'sigma_u', 'sigma_c', 'sigma_H', 'sigma_Hdot', 'sigma_frc', 'sigma_e', 'gamma',
                          'passage_times', 'saccade_times', 'Delta']

    assert all([param_name in params_all for param_name in params_to_optimize]), "ERROR: Some parameter cannot be used for this control method."
    params_to_optimize_info = [(param, params_info[param][-1]) for param in params_info
                                     for _ in range(params_info[param][2] if params_info[param][-1] in ['discrete_sequence',
                                                                            'continuous_sequence'] else 1)
                                     if param in params_to_optimize]
    boundaries = [params_info[param_name][:2] for param_name, param_type in params_to_optimize_info]

    # 4. Run optimization with Differential Evolution
    control_method_name = system_dynamics if control_method == "LQG" else control_method  # ensure that "E-LQG" is printed instead of "LQG"
    print(f"Computing parameters\n{params_to_optimize}\nthat minimize {loss_type} between {control_method_name} and user data\n"
          f"(User: {user}, Distance: {distance}, Width: {width}, Direction: {direction}).")
    res = differential_evolution(paramfitting_loss, args=(
        params_to_optimize_info, param_dict_fixed,
        Nmax, dt, x0_mean, u0, Sigma0, dim, num_targets, analysis_dim,
        x_loc_data, x_scale_data,
        control_method, system_dynamics, loss_type, fitting_method, n_simulation_samples,
        include_proprioceptive_target_signals, include_proprioceptive_endeffector_signals,
        modify_init_target_estimate, use_square_root_filter,
        min_algorithm_iterations, algorithm_iterations, J_eps), bounds=boundaries, polish=False, updating="deferred", workers=-1)

    # 5. Set up optimal param dict
    param_dict_optimize = {}
    for key, value in zip(params_to_optimize_info, res.x):
        if key[1] in ['discrete_sequence', 'continuous_sequence']:
            if key[0] == 'passage_times':
                param_dict_optimize.setdefault(key[0], [0]).append(
                    value)  # first "target switch" (i.e., from initial position to first real target) is fixed at time zero
            elif key[0] == 'saccade_times':
                param_dict_optimize.setdefault(key[0], [Nmax]).insert(0, value)  # last "saccade time" (i.e., from last target to last target...) is fixed at time N
            else:
                param_dict_optimize.setdefault(key[0], []).append(value)
        else:
            param_dict_optimize[key[0]] = value
    print(f"\nOptimal {control_method_name} parameters (User: {user}, Distance: {distance}, Width: {width}, Direction: {direction}):\n"
          f"*** {loss_type}={res.fun} ***\n"
          f"{param_dict_optimize}")

    return res

def paramfitting_loss(param_flattened, param_info, param_dict_fixed,
                      N, dt, x0_mean, u0, Sigma0, dim, num_targets,
                      analysis_dim, x_loc_data, x_scale_data=None,
                      control_method="LQG",
                      system_dynamics="LQG",
                      loss_type="MWD",
                      fitting_method="expectation-based",
                      n_simulation_samples=200,
                      include_proprioceptive_target_signals=False,
                      include_proprioceptive_endeffector_signals=False,
                      modify_init_target_estimate=True,  #only used for E-LQG
                      use_square_root_filter=True,
                      min_algorithm_iterations=1, algorithm_iterations=20, J_eps=1e-3):
    """
    Computes loss based on error between simulation trajectory and reference data trajectory, given some parameter set.
    :param param_flattened: list of parameter values (main input argument) [list of floats]
    :param param_info: list of information tuples for each parameter value consisting of parameter name and parameter type [list of tuples]
    :param param_dict_fixed: dictionary with name-value pairs of remaining (fixed) parameters; missing parameters are set to their default value [dict]
    :param N: number of time steps (excluding final step) [int]
    :param dt: time step duration in seconds [h in paper] [int/float]
    :param x0_mean: initial state (expectation) ((flattened) position, velocity, muscle force,
    muscle excitation, [initial position (only if control_method=="LQG" and system_dynamics=="E-LQG"),] target position(s))
    (corresponds to x0 if control_method=="LQR") [array-like (1D)]
    :param u0: initial control value (unused here) [dim-array]
    :param Sigma0: initial state covariance matrix (only used if control_method=="LQG") [array-like (2D)]
    :param dim: dimension of the task (1D, 2D, or 3D) [int]
    :param num_targets: number of targets [if control_method=="LQG" and system_dynamics=="E-LQG": including initial position]
    (can be used for via-point tasks) [int]
    :param analysis_dim: how many dimensions of user data trajectories should be considered in loss function (e.g., 2 means position and velocity only) [1-3]
    :param x_loc_data: average user trajectory [(N+1)*analysis_dim-array]
    :param x_scale_data: covariance sequence of user trajectory [(N+1)*analysis_dim*analysis_dim-array]
    :param control_method: control method to use ["2OL-Eq", "MinJerk", "LQR", or "LQG"]
    :param system_dynamics: which dynamics to use (only used if control_method=="LQG") ["LQG" or "E-LQG"]
    :param loss_type: loss function to use ["SSE", "Maximum Error", "MAE", "MKL", or "MWD" (availability also depends on "control_method")]
    :param fitting_method: whether to compare simulation and user data based on expectation or sample statistics ["expectation-based" or "sample-based"]
    :param n_simulation_samples: number of simulation samples used to compute distribution statistics (only if fitting_method=="sample-based") [int]
    :param include_proprioceptive_target_signals: whether target position(s) can be observed in absolute coordinates
    (only usable for LQG; default: False) [bool]
    :param include_proprioceptive_endeffector_signals: whether end-effector can be observed position in absolute coordinates
    (only usable for E-LQG; default: False) [bool]
    :param modify_init_target_estimate: whether to place an incorrect initial target estimate
    (basically, the target estimate is set to initial movement position, which makes particular sense for via-point tasks)
    (only works for E-LQG or if "include_proprioceptive_target_signals==True") [bool]
    :param use_square_root_filter: whether to use the square root filter to update Kalman matrices (default: True) [bool]
    :param min_algorithm_iterations: minimum number of iterations (see "algorithm_iterations")
    :param algorithm_iterations: (maximum) number of iterations, where the optimal control and
    the optimal observation problem is solved alternately (if "J_eps" is set, early termination is possible) [int]
    :param J_eps: if relative improvement of cost function falls below "J_eps" and "min_algorithm_iterations" is reached,
    iterative solving algorithm terminates [float]
    :return: loss [float]
    """

    # 1. Set up param dict
    param_dict_optimize = {}
    for key, value in zip(param_info, param_flattened):
        if key[1] in ['discrete_sequence', 'continuous_sequence']:
            if key[0] == 'passage_times':
                param_dict_optimize.setdefault(key[0], [0]).append(
                    value)  # first "target switch" (i.e., from initial position to first real target) is fixed at time zero
            elif key[0] == 'saccade_times':
                param_dict_optimize.setdefault(key[0], [N]).insert(0, value)  # last "saccade time" (i.e., from last target to last target...) is fixed at time N
            else:
                param_dict_optimize.setdefault(key[0], []).append(value)
        else:
            param_dict_optimize[key[0]] = value
    param_dict_complete = {**param_dict_fixed, **param_dict_optimize}  #WARNING: order might play a role

    # 2. Compute simulation trajectory
    if control_method in ["LQG", "E-LQG"]:
        if fitting_method == "expectation-based":
            (_, x_expectation, Sigma_x, _, _, _, _, _, _, _) = lqg(N, dt, x0_mean, u0, Sigma0, dim, num_targets,
                                                                system_dynamics=system_dynamics,
                                                                include_proprioceptive_target_signals=include_proprioceptive_target_signals,
                                                                include_proprioceptive_endeffector_signals=include_proprioceptive_endeffector_signals,
                                                                modify_init_target_estimate=modify_init_target_estimate,
                                                                minimum_computations=True,
                                                                min_algorithm_iterations=min_algorithm_iterations,
                                                                algorithm_iterations=algorithm_iterations,
                                                                J_eps=J_eps,
                                                                **param_dict_complete)

            x_loc_sim = x_expectation[:, :analysis_dim]
            x_scale_sim = np.squeeze([cov_matrix[:analysis_dim, :analysis_dim] for cov_matrix in Sigma_x])
        elif fitting_method == "sample-based":
            x_SIMULATION = []
            Delta = param_dict_complete["Delta"] if "Delta" in param_dict_complete else 0

            system_matrices, n, m, l, gamma, passage_times, saccade_times = lqg_initialization(N, dt, dim,
                                                                                               system_dynamics=system_dynamics,
                                                                                               num_targets=num_targets,
                                                                                               include_proprioceptive_endeffector_signals=include_proprioceptive_endeffector_signals,
                                                                                               include_proprioceptive_target_signals=include_proprioceptive_target_signals,
                                                                                               **param_dict_complete)

            initial_state_matrices = lqg_setup_initial_state(dim, num_targets, x0_mean, u0, Sigma0, n, m, Delta=Delta,
                                                             system_dynamics=system_dynamics,
                                                             include_proprioceptive_target_signals=include_proprioceptive_target_signals,
                                                             modify_init_target_estimate=modify_init_target_estimate)

            Ical0, Sigma_x0 = initial_state_matrices[:2]
            J = 10e+12  #use very large costs costs as initial reference value
            Kk = None

            # Alternately solve the optimal control given fixed Kalman gain matrices Kk and the optimal observation problem given fixed feedback gains L
            for iter in range(algorithm_iterations + 1):

                # Solve the optimal control problem given fixed Kalman gain matrices Kk and compute resulting costs
                J_before = J
                L, S_x, S_e, alpha = lqg_solve_control_problem(N, dim, num_targets, gamma, saccade_times,
                                                               system_matrices, m, l,
                                                               Delta=Delta, Kk=Kk)
                J = lqg_compute_optimal_costs(Ical0, Sigma_x0, S_x, S_e, alpha)
                # assert J >= 0
                if J < 0:
                    print(f"!!! IMPORTANT WARNING (LQG): J = {J}!")


                # Solve the optimal observation problem given fixed feedback gains L
                Kk, Kcal, Sigma_Ical, Sigma_ecal, Sigma_Icalecal = lqg_solve_observation_problem(N, dim, num_targets,
                                                                                                 gamma,
                                                                                                 saccade_times,
                                                                                                 system_matrices,
                                                                                                 initial_state_matrices,
                                                                                                 m, l, Delta=Delta, L=L,
                                                                                                 use_square_root_filter=use_square_root_filter)


                # Decide whether to stop iteration
                if J_eps is not None:
                    if (J <= J_before) and (np.abs((J - J_before) / J_before) <= J_eps) and (
                            iter >= min_algorithm_iterations - 1):
                        #print(f'{system_dynamics} converged after {iter + 1} iterations.')
                        break
                    elif iter == algorithm_iterations - 1:
                        print(f'WARNING: LQG stopped after {iter + 1} iterations without convergence.')

            for _ in range(n_simulation_samples):  # run multiple simulations
                (_, _, _, _, _, _, x_SIMULATION_sample, _, _) = lqg_forward_simulation(
                                                                        L,
                                                                        Kk, Kcal, Sigma_Ical, Sigma_ecal, Sigma_Icalecal,
                                                                        N,
                                                                        dim,
                                                                        num_targets,
                                                                        x0_mean, u0, Sigma0,
                                                                        system_matrices,
                                                                        initial_state_matrices,
                                                                        n, m, l,
                                                                        gamma, saccade_times,
                                                                        Delta=Delta,
                                                                        minimum_computations=True)
                x_SIMULATION.append(x_SIMULATION_sample)  # extract position and velocity from simulation data
            x_loc_sim = np.mean(x_SIMULATION, axis=0)[:, :analysis_dim]
            x_scale_sim = np.squeeze([np.cov([x_SIMULATION_sample[i, :analysis_dim] for x_SIMULATION_sample in x_SIMULATION], rowvar=False)
                                  for i in range(N + 1)])
        else:
            raise NotImplementedError
    elif control_method == "LQR":
        (_, x, _) = lqr(N, dt, x0_mean, dim, num_targets=num_targets, **param_dict_complete)

        x_loc_sim = x[:, :analysis_dim]
    elif control_method == "MinJerk":
        T = x0_mean[-1*dim:]  # extract target position from last component
        x0_mean = x0_mean[:3*dim]  # remove target(s) from state space, since they have no effect in MinJerk
        (x, _) = minjerk(N, dt, x0_mean, dim, T, **param_dict_complete)

        x_loc_sim = x[:, :analysis_dim]
    elif control_method == "2OL-Eq":
        T = x0_mean[-1*dim:]  # extract target position from last component
        x0_mean = x0_mean[:2*dim]  # remove target(s) from state space, since they have no effect in 2OL-Eq
        (x, _) = secondorderlag_eq(N, dt, x0_mean, dim, T, **param_dict_complete)

        x_loc_sim = x[:, :analysis_dim]
    else:
        raise NotImplementedError

    # 3. Compute metrics that compare simulation and user data
    if (control_method == "LQG") and (system_dynamics == "E-LQG"):
        control_method = "E-LQG"  #ensure that "E-LQG" is printed instead of "LQG"
    if loss_type == "SSE":
        loss = compute_SSE(x_loc_sim, x_loc_data)
        print(f"{control_method} - SSE ({param_dict_optimize}): {loss}")
    elif loss_type == "Maximum Error":
        loss = compute_MaximumError(x_loc_sim, x_loc_data)
        print(f"{control_method} - Maximum Error ({param_dict_optimize}): {loss}")
    elif loss_type == "MAE":
        loss = compute_MAE(x_loc_sim, x_loc_data)
        print(f"{control_method} - MAE ({param_dict_optimize}): {loss}")
    elif loss_type == "MKL":
        assert control_method in ["LQG", "E-LQG"], f"The loss type {loss_type} cannot be used with {control_method}."
        loss = compute_MKL_normal(x_loc_data, x_scale_data, x_loc_sim, x_scale_sim)  #WARNING: order of arguments matter, since MKL is no real metric!
        print(f"{control_method} - Mean KL Divergence ({param_dict_optimize}): {loss}")
    elif loss_type == "MWD":
        assert control_method in ["LQG", "E-LQG"], f"The loss type {loss_type} cannot be used with {control_method}."
        loss = compute_MWD_normal(x_loc_data, x_scale_data, x_loc_sim, x_scale_sim)
        print(f"{control_method} - Mean Wasserstein Distance ({param_dict_optimize}): {loss}")
    else:
        raise NotImplementedError

    return loss


def compute_SSE(x, y, weights=1):
    """
    Computes Sum Squared Error (SSE) between two trajectories x and y.
    :param x: first trajectory to compare [array-like (2D)]
    :param y: second trajectory to compare [array-like (2D)]
    :param weights: weights for dimensions [int/float or array-like (1D)]
    :return: Sum Squared Error (SSE) [float]
    """

    x = np.squeeze(x)
    y = np.squeeze(y)
    assert x.shape == y.shape, "Dimension of trajectories does not match!"

    res = x - y
    weighted_res = weights * res  # weights should be a 1D numpy array in general

    SSE = np.sum(weighted_res ** 2)

    return SSE


def compute_MaximumError(x, y, weights=1):
    """
    Computes Maximum (Absolute) Error between two trajectories x and y.
    :param x: first trajectory to compare [array-like (2D)]
    :param y: second trajectory to compare [array-like (2D)]
    :param weights: weights for dimensions [int/float or array-like (1D)]
    :return: Maximum (Absolute) Error [float]
    """

    x = np.squeeze(x)
    y = np.squeeze(y)
    assert x.shape == y.shape, "Dimension of trajectories does not match!"

    res = x - y
    weighted_res = weights * res  # weights should be a 1D numpy array in general

    maxerror = np.max(abs(weighted_res))

    return maxerror


def compute_MAE(x, y, weights=1):
    """
    Computes Mean Absolute Error (MAE) between two trajectories x and y.
    :param x: first trajectory to compare [array-like (2D)]
    :param y: second trajectory to compare [array-like (2D)]
    :param weights: weights for dimensions [int/float or array-like (1D)]
    :return: Mean Absolute Error (MAE) [float]
    """

    x = np.squeeze(x)
    y = np.squeeze(y)
    assert x.shape == y.shape, "Dimension of trajectories does not match!"

    res = x - y
    weighted_res = weights * res  # weights should be a 1D numpy array in general

    if len(np.squeeze(x).shape) == 1:  # only position was given
        MAE = np.mean(abs(weighted_res))
    else:
        MAE = np.mean(np.mean(abs(weighted_res), axis=0))
        raise NotImplementedError  # check whether outcome is as desired (mean of MAEs?)

    return MAE


def compute_MKL_normal(p_mean, p_scale, q_mean, q_scale):
    """
    Computes the mean of the KL divergences KL(pi||qi) for given sequences of normal distributions q=(qi) and p=(pi).
    For formula derivation, see, e.g, https://stats.stackexchange.com/a/281725.
    :param p_mean: means of distributions pi [array-like (1D/2D), with first dimension corresponding to distributions]
    :param p_scale: covariances of distributions pi [array-like (2D/3D), with first dimension corresponding to distributions]
    :param q_mean: means of distributions qi [array-like (1D/2D), with first dimension corresponding to distributions]
    :param q_scale: covariances of distributions qi [array-like (2D/3D), with first dimension corresponding to distributions]
    :return: mean(KL(pi||qi))
    """

    p_mean = np.squeeze(p_mean)
    p_scale = np.squeeze(p_scale)
    q_mean = np.squeeze(q_mean)
    q_scale = np.squeeze(q_scale)
    assert p_mean.shape == q_mean.shape
    assert p_scale.shape == q_scale.shape
    if len(p_scale.shape) == 3:
        assert p_scale.shape[1] == p_scale.shape[2]
    N = p_scale.shape[0] - 1
    data_dim = p_scale.shape[1]

    kl_divergence_sequence = np.zeros(shape=(N + 1,))
    for i in range(N + 1):
        pt_1 = np.log(np.linalg.det(q_scale[i]) / np.linalg.det(p_scale[i])) - data_dim
        # assert not np.isnan(pt_1), i
        if np.isnan(pt_1):
            print(
                f"!!! IMPORTANT WARNING: KL_divergence at step i={i} is NaN (probably because p_scale[{i}] has negative eigenvalues, i.e., det(p_scale[{i}]={np.linalg.det(q_scale[i])})!")
        q_scale_inv = np.linalg.pinv(q_scale[i])
        pt_2 = np.trace(q_scale_inv.dot(p_scale[i])) + (q_mean[i] - p_mean[i]).dot(q_scale_inv).dot(
            q_mean[i] - p_mean[i])
        kl_divergence_sequence[i] = 0.5 * (pt_1 + pt_2)
    mean_kl_divergence = kl_divergence_sequence.mean()

    return mean_kl_divergence


def compute_MWD_normal(p_mean, p_scale, q_mean, q_scale):
    """
    Computes the mean of the (Euclidean) Wasserstein distance W_2(pi,qi) = W_2(qi,pi)
    for given sequences of normal distributions q=(qi) and p=(pi).
    For formula derivation, see, e.g, https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions.
    :param p_mean: means of distributions pi [array-like (1D/2D), with first dimension corresponding to distributions]
    :param p_scale: covariances of distributions pi [array-like (2D/3D), with first dimension corresponding to distributions]
    :param q_mean: means of distributions qi [array-like (1D/2D), with first dimension corresponding to distributions]
    :param q_scale: covariances of distributions qi [array-like (2D/3D), with first dimension corresponding to distributions]
    :return: mean(W_2(pi,qi))
    """

    p_mean = np.squeeze(p_mean)
    p_scale = np.squeeze(p_scale)
    q_mean = np.squeeze(q_mean)
    q_scale = np.squeeze(q_scale)
    assert p_mean.shape == q_mean.shape
    assert p_scale.shape == q_scale.shape
    if len(p_scale.shape) == 3:
        assert p_scale.shape[1] == p_scale.shape[2]
    N = p_scale.shape[0] - 1

    wasserstein_distance_sequence = np.zeros(shape=(N + 1,))

    for i in range(N + 1):
        ### W_2(p, q) [Wasserstein metric]:
        pt_1 = np.linalg.norm(q_mean[i] - p_mean[i]) ** 2
        pt_2 = np.trace(q_scale[i]) + np.trace(p_scale[i]) - 2 * np.trace(sqrtm(q_scale[i].dot(p_scale[i])))
        if np.abs(pt_2) <= 10e-16:
            pt_2 = 0
        wasserstein_distance_sequence[i] = np.sqrt(pt_1 + pt_2)

    mean_wasserstein_distance = wasserstein_distance_sequence.mean()

    return mean_wasserstein_distance


def sqrtm_psd(A, est_error=False, check_finite=True):
    """
    Returns the matrix square root of a positive semidefinite matrix,
    truncating negative eigenvalues.

    Copyright 2017, Chris Ferrie and Christopher Granade.
    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
    promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    :param A: input matrix [array-like (2D)]
    :param est_error: whether to return error [bool]
    :param check_finite: whether to check that the input matrices contain only finite numbers [bool]
    :return:
        - A_sqrt: matrix square root [2D-array]
        [- error between A_sqrt.dot(A_sqrt) and A in Frobenius norm (only if "est_error==True")]
    """

    w, v = eigh(A, check_finite=check_finite)
    mask = w <= 0
    w[mask] = 0
    np.sqrt(w, out=w)
    A_sqrt = (v * w).dot(v.conj().T)

    if est_error:
        return A_sqrt, np.linalg.norm(np.dot(A_sqrt, A_sqrt) - A, 'fro')
    else:
        return A_sqrt

