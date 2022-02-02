from ast import literal_eval

import ofc4hci

import argparse

parser = argparse.ArgumentParser(description='Provide some command line options.', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('control_method', type=str,
                    help='Control method used for parameter optimization. '
                         'Valid control models are: "2OL-Eq", "MinJerk", "LQR", "LQG", and "E-LQG".')
parser.add_argument('params_to_optimize', type=str, nargs='+',
                    help='Parameters of chosen control model that should be optimized.\n'
                         'Possible params of...\n'
                         '- 2OL-Eq: "k", "d"\n'
                         '- MinJerk: "passage_times"\n'
                         '- LQR: "r", "velocitycosts_weight", "forcecosts_weight", "mass", "t_const_1", "t_const_2"\n'
                         '- LQG: "r", "velocitycosts_weight", "forcecosts_weight", "mass", "t_const_1", "t_const_2", '
                         '"sigma_u", "sigma_c", "sigma_s", "passage_times", "Delta"\n'
                         '- E-LQG: "r", "velocitycosts_weight", "forcecosts_weight", "mass", "t_const_1", "t_const_2", '
                         '"sigma_u", "sigma_c", "sigma_H", "sigma_Hdot", "sigma_frc", "sigma_e", "gamma", '
                         '"passage_times", "saccade_times", "Delta"')

parser.add_argument('--user', type=int, default=3,  help='User ID (1-12).')
parser.add_argument('--distance', type=int, default=765,  help='Distance to target in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}].')
parser.add_argument('--width', type=int, default=51,  help='Target width in px [(distance, width) must be from {(765, 255), (1275, 425), (765, 51), (1275, 85), (765, 12), (1275, 20), (765, 3), (1275, 5)}].')
parser.add_argument('--direction', type=str, default='right',  help='Movement direction ["left" or "right"].')

parser.add_argument('--params_fixed',
                    type=lambda kv_pairs: {k: literal_eval(v) for k, v in [kv_pair.split('=') for kv_pair in kv_pairs.split()]}, default={},
                    help='"name=value"-pairs of remaining (fixed) parameters, split by whitespaces.\n'
                         'Missing parameters are set to their default or optimal value.')
parser.add_argument('--loss_type',
                    type=str, default='SSE',
                    help='Loss function to use ("SSE", "Maximum Error", "MAE", "MKL", or "MWD"; availability might depend on "control_method").')

if __name__=="__main__":

    args = parser.parse_args()
    user, distance, width, direction = args.user, args.distance, args.width, args.direction
    control_method, params_to_optimize, loss_type, param_dict_fixed = args.control_method, args.params_to_optimize, args.loss_type, args.params_fixed

    target_dict = {(765, 255): 2, (1275, 425): 2, (765, 51): 4, (1275, 85): 4, (765, 12): 6, (1275, 20): 6, (765, 3): 8,
                   (1275, 5): 8}
    print(f"Run parameter optimization for User {user}, ID {target_dict[(distance, width)]} (distance: {distance}, width: {width}), {direction} movements.\n"
          f"Control method:  {control_method}\n"
          f"Parameters: {params_to_optimize}\n"
          f"Loss function: {loss_type}\n"
          f"{'Other non-default parameter values: {}'.format(param_dict_fixed) if len(param_dict_fixed) > 0 else ''}")

    if control_method == "E-LQG":  #E-LQG is implemented as variant of LQG with different system dynamics
        control_method = "LQG"
        system_dynamics = "E-LQG"
    else:
        system_dynamics = "LQG"

    ofc4hci.paramfitting(user, distance, width, direction, params_to_optimize, param_dict_fixed,
                         control_method=control_method, system_dynamics=system_dynamics, loss_type=loss_type)
