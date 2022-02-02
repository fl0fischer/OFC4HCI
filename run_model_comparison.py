import pandas as pd
from ast import literal_eval

import ofc4hci

import argparse

parser = argparse.ArgumentParser(description='Provide some command line options.', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('control_methods', type=str, nargs='+',
                    help='Control methods included in comparison (multiple methods split by whitespaces). '
                         'Valid control methods are: "2OL-Eq", "MinJerk", "LQR", "LQG", and "E-LQG".')

parser.add_argument('--user', type=int, default=3, help='User ID (1-12).')
parser.add_argument('--distance', type=int, default=765, help='Distance to target in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}].')
parser.add_argument('--width', type=int, default=51, help='Target width in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}].')
parser.add_argument('--direction', type=str, default='right', help='Movement direction ["left" or "right"].')

parser.add_argument('--no-opt', dest='use_opt_params', action='store_false', default=True,
                    help='If True, default values are used instead of optimal parameter values for non-specified parameters.')
parser.add_argument('--secondorderlag_eq_params',
                    type=lambda kv_pairs: {k: literal_eval(v) for k, v in [kv_pair.split('=') for kv_pair in kv_pairs.split()]}, default={},
                    help='"name=value"-pairs of 2OL-Eq parameters, split by whitespaces.\n'
                         'Possible parameters: "k", "d".\n'
                         'Missing parameters are set to their default or optimal value.')
parser.add_argument('--minjerk_params',
                    type=lambda kv_pairs: {k: literal_eval(v) for k, v in [kv_pair.split('=') for kv_pair in kv_pairs.split()]}, default={},
                    help='"name=value"-pairs of MinJerk parameters, split by whitespaces.\n'
                         'Possible parameters: "passage_times".\n'
                         'Missing parameters are set to their default or optimal value.')
parser.add_argument('--lqr_params',
                    type=lambda kv_pairs: {k: literal_eval(v) for k, v in [kv_pair.split('=') for kv_pair in kv_pairs.split()]}, default={},
                    help='"name=value"-pairs of LQR parameters, split by whitespaces.\n'
                         'Possible parameters: "r", "velocitycosts_weight", "forcecosts_weight", "mass", "t_const_1", "t_const_2".\n'
                         'Missing parameters are set to their default or optimal value.')
parser.add_argument('--lqg_params',
                    type=lambda kv_pairs: {k: literal_eval(v) for k, v in [kv_pair.split('=') for kv_pair in kv_pairs.split()]}, default={},
                    help='"name=value"-pairs of LQG parameters, split by whitespaces.\n'
                         'Possible parameters: "r", "velocitycosts_weight", "forcecosts_weight", "mass", "t_const_1", "t_const_2", '
                         '"sigma_u", "sigma_c", "sigma_s", "passage_times", "Delta".\n'
                         'Missing parameters are set to their default or optimal value.')
parser.add_argument('--elqg_params',
                    type=lambda kv_pairs: {k: literal_eval(v) for k, v in [kv_pair.split('=') for kv_pair in kv_pairs.split()]}, default={},
                    help='"name=value"-pairs of E-LQG parameters, split by whitespaces.\n'
                         'Possible parameters: "r", "velocitycosts_weight", "forcecosts_weight", "mass", "t_const_1", "t_const_2", '
                         '"sigma_u", "sigma_c", "sigma_H", "sigma_Hdot", "sigma_frc", "sigma_e", "gamma", '
                         '"passage_times", "saccade_times", "Delta".\n'
                         'Missing parameters are set to their default or optimal value.')


if __name__=="__main__":

    args = parser.parse_args()
    user, distance, width, direction = args.user, args.distance, args.width, args.direction

    target_dict = {(765, 255): 2, (1275, 425): 2, (765, 51): 4, (1275, 85): 4, (765, 12): 6, (1275, 20): 6, (765, 3): 8,
                   (1275, 5): 8}
    print(f"Run model comparison for User {user}, ID {target_dict[(distance, width)]} (distance: {distance}, width: {width}), {direction} movements.")

    valid_control_methods = ["2OL-Eq", "MinJerk", "LQR", "LQG", "E-LQG"]
    assert set(args.control_methods) <= set(valid_control_methods), f"Invalid control method names have been passed.\nPassed control methods: {args.control_methods}\nValid control methods: {valid_control_methods}."

    # Compute trajectories and store metrics for each desired model:
    metrics = []
    if "2OL-Eq" in args.control_methods:
        control_method = "2OL-Eq"

        param_dict_custom = args.secondorderlag_eq_params
        param_dict = ofc4hci._get_custom_param_dict(control_method, user, distance, width, direction,
                                                    param_dict_custom=param_dict_custom, use_opt_params=args.use_opt_params)

        xopt, uopt, x_loc_data, SSE, MaximumError = ofc4hci.secondorderlag_eq_pointingdynamics(user, distance, width, direction,
                                                                                           secondorderlag_eq_param_dict=param_dict)

        metrics.append({"MODEL": control_method, "SSE": SSE, "Maximum Error": MaximumError})
    if "MinJerk" in args.control_methods:
        control_method = "MinJerk"

        param_dict_custom = args.minjerk_params
        param_dict = ofc4hci._get_custom_param_dict(control_method, user, distance, width, direction,
                                                    param_dict_custom=param_dict_custom, use_opt_params=args.use_opt_params)

        xopt, uopt, x_loc_data, SSE, MaximumError = ofc4hci.minjerk_pointingdynamics(user, distance, width, direction,
                                                       minjerk_param_dict=param_dict)
        metrics.append({"MODEL": control_method, "SSE": SSE, "Maximum Error": MaximumError})
    if "LQR" in args.control_methods:
        control_method = "LQR"

        param_dict_custom = args.lqr_params
        param_dict = ofc4hci._get_custom_param_dict(control_method, user, distance, width, direction,
                                                    param_dict_custom=param_dict_custom, use_opt_params=args.use_opt_params)

        Jopt, xopt, uopt, x_loc_data, SSE, MaximumError = ofc4hci.lqr_pointingdynamics(user, distance, width, direction,
                                                                                       lqr_param_dict=param_dict)
        metrics.append({"MODEL": control_method, "SSE": SSE, "Maximum Error": MaximumError})
    if "LQG" in args.control_methods:
        control_method = "LQG"

        param_dict_custom = args.lqg_params
        param_dict = ofc4hci._get_custom_param_dict(control_method, user, distance, width, direction,
                                                    param_dict_custom=param_dict_custom, use_opt_params=args.use_opt_params)

        Ical_expectation, Sigma_x, x_loc_data, x_scale_data, \
        SSE, MaximumError, MKL, MWD = ofc4hci.lqg_pointingdynamics(user, distance, width, direction, system_dynamics="LQG",
                                                                   lqg_param_dict=param_dict)
        metrics.append({"MODEL": control_method, "SSE": SSE, "Maximum Error": MaximumError, "MKL": MKL, "MWD": MWD})
    if "E-LQG" in args.control_methods:
        control_method = "E-LQG"

        param_dict_custom = args.elqg_params
        param_dict = ofc4hci._get_custom_param_dict(control_method, user, distance, width, direction,
                                                    param_dict_custom=param_dict_custom, use_opt_params=args.use_opt_params)

        Ical_expectation, Sigma_x, x_loc_data, x_scale_data, \
        SSE, MaximumError, MKL, MWD = ofc4hci.lqg_pointingdynamics(user, distance, width, direction, system_dynamics="E-LQG", lqg_param_dict=param_dict)
        metrics.append({"MODEL": control_method, "SSE": SSE, "Maximum Error": MaximumError, "MKL": MKL, "MWD": MWD})

    df = pd.DataFrame(metrics).set_index("MODEL")
    print(df)
