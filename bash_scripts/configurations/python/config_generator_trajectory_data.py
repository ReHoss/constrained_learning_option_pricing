# Create a generator of config files for python
import argparse
import datetime
import itertools
import pathlib

import yaml


"""
1 - Do not forget to change the mlfow_experiment_name value
2 - The file naming may impact file generation when overwriting is permitted, an
    exception has been included to prevent this
"""

text_yaml_config_file = f"""
seed: 94
mlflow_experiment_name: {datetime.datetime.now().strftime('%Y_%m_%d')}_training_data

data:
  type: unique_trajectory_divided
  num_steps: 200
  control_type: uniform
  num_trajectories: 400


env:
  name: pendulum
  params:
    dt: 0.05
    delay_observation: 0.0
"""

# Default target directory with the date and time
# noinspection DuplicatedCode
name_target_directory = (
    f"{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_generated_configs"
)

# Argparse the name of the target directory with flag -d or --directory (optional)
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help="Name of the target directory")
args = parser.parse_args()
name_target_directory = args.directory if args.directory else name_target_directory

# Get current file directory with Pathlib
path_parent_directory = pathlib.Path(__file__).parent

path_target_directory = pathlib.Path(
    f"{path_parent_directory}/../../../config/batch/{name_target_directory}"
).resolve()
# Check if the target directory exists
if not pathlib.Path(pathlib.Path(path_target_directory).resolve()).exists():
    # Create the target directory
    pathlib.Path(path_target_directory).mkdir(parents=False, exist_ok=False)

# Load the yaml text as a dictionary
dict_config = yaml.load(text_yaml_config_file, Loader=yaml.FullLoader)

# Add a suffix to the mlflow_experiment_name to differentiate the runs
current_time = datetime.datetime.now().strftime("%Hh_%Mm")
dict_config["mlflow_experiment_name"] += f"_{current_time}"

# --- Start of the list of parameters to change
# Define the list of parameters to change,
# by giving in the tuple the nested keys of the dictionary


N_SEEDS = 1
DICT_ENV_PARAMS = [
    # {
    #     "name": "pendulum",
    #     "params": {
    #         "delay_observation": 0.0,
    #         "dt": 0.1,
    #         "dict_initial_condition": {"type": "bottom", "std": 0.1},
    #     },
    # },
    {
        "name": "mackey_glass",
        "params": {
            "delay_observation": 0.0,
            "dt": 0.1,
            "gamma": 1.0,
            "beta": 2.0,
            "tau": 3.0,
            "n": 7.0,
            "dict_initial_condition": {"type": "nonzero", "std": 0.01},
            "max_control": 0.0,
        },
    },
    # {
    #     "name": "vanderpol",
    #     "params": {
    #         "delay_observation": 0.0,
    #         "dt": 0.1,
    #         "dict_initial_condition": {"type": "zero", "std": 0.1},
    #     },
    # },
    # {
    #     "name": "cavity",
    #     "params": {
    #         "name_flow": "cavity",
    #         "delay_observation": 0.0,
    #         "max_control": None,
    #         "dt": 0.01,
    #         "reynolds": 10.0,
    #         "mesh": "coarse",
    #         "dict_initial_condition": {"type": "ergodic", "std": 0.1},
    #         "dict_solver": {
    #             "name": "semi_implicit_bdf",
    #             "dt": 0.001,
    #             "order": 2,
    #             "stabilization": "none",
    #         },
    #         "paraview_callback_interval": 1000,
    #         "log_callback_interval": 1000,
    #     },
    # },
    # {
    #     "name": "cylinder",
    #     "params": {
    #         "name_flow": "cylinder",
    #         "delay_observation": 0.0,
    #         "max_control": None,
    #         "dt": 0.01,
    #         "reynolds": 30.0,
    #         "mesh": "coarse",
    #         "dict_initial_condition": {"type": "ergodic", "std": 0.1},
    #         "dict_solver": {
    #             "name": "semi_implicit_bdf",
    #             "dt": 0.01,
    #             "order": 2,
    #             "stabilization": "none",
    #         },
    #         "paraview_callback_interval": 100,
    #         "log_callback_interval": 100,
    #     },
    # },
    # {
    #     "name": "pinball",
    #     "params": {
    #         "name_flow": "pinball",
    #         "delay_observation": 0.0,
    #         "max_control": None,
    #         "dt": 0.01,
    #         "reynolds": 30.0,
    #         "mesh": "coarse",
    #         "dict_initial_condition": {"type": "ergodic", "std": 0.1},
    #         "dict_solver": {
    #             "name": "semi_implicit_bdf",
    #             "dt": 0.01,
    #             "order": 2,
    #             "stabilization": "none",
    #         },
    #         "paraview_callback_interval": 100,
    #         "log_callback_interval": 100,
    #     },
    # },
]


LIST_DATA_TYPES = ["unique_trajectory_divided", "multiple_trajectories"]
LIST_DELAY_OBSERVATION = [0.0]
LIST_CONTROL_TYPES = ["uniform", "harmonic_process"]
# LIST_MAX_CONTROL = [0.0, 0.1]
# DICT_LIST_REYNOLDS = {
#     "cylinder": [50, 90, 105, 120],
#     "pinball": [50, 90, 105, 120],
#     "cavity": [500, 5000, 7500],
# }
DICT_LIST_MAX_CONTROL = {
    # "cylinder": [0.0, 0.01],
    # "pinball": [0.0, 0.01],
    # "cavity": [0.0, 0.01],
    "mackey_glass": [0.0, 0.01],
}

# DICT_N_TRAJECTORIES = {
#     "cylinder": 400,
#     "pinball": 400,
#     "cavity": 40,
# }

list_env_params = [
    (
        "env",
        {
            "name": dict_env["name"],
            "params": {
                **dict_env["params"],
                "name_flow": dict_env["name"],
                # "reynolds": reynolds,
                # "delay_observation": delay_observation,
                "max_control": max_control,
            },
        },
    )
    for dict_env in DICT_ENV_PARAMS
    # for delay_observation in LIST_DELAY_OBSERVATION
    for max_control in DICT_LIST_MAX_CONTROL[dict_env["name"]]
    # for reynolds in DICT_LIST_REYNOLDS[dict_env["name"]]
]


list_data_types = [
    (
        "data",
        {
            **dict_config["data"],
            "type": data_type,
            "control_type": control_type,
            # "num_trajectories": n_trajectories
        },
    )
    for data_type in LIST_DATA_TYPES
    for control_type in LIST_CONTROL_TYPES
    # for n_trajectories in set(DICT_N_TRAJECTORIES.values())
]

list_seed = [("seed", seed) for seed in range(N_SEEDS)]

# !!!! UPDATE THE NESTED LIST OF PARAMETERS TO CHANGE HERE !!!!
nested_list_parameters = [list_seed, list_env_params, list_data_types]

# --- End of the list of parameters to change


list_itertool_product = list(itertools.product(*nested_list_parameters))


# Filter out the combinations that are not valid
# Here keep only the combinations where the number of trajectories matches
# what is specified in the dictionary DICT_N_TRAJECTORIES

# list_itertool_product = list(
#     filter(
#         lambda _tuple_parameters: DICT_N_TRAJECTORIES[_tuple_parameters[1][1]["name"]]
#         == _tuple_parameters[2][1]["num_trajectories"],
#         list_itertool_product,
#     )
# )


for tuple_parameters in list_itertool_product:
    # Create a new dictionary with the new parameters
    dict_config_new = dict_config.copy()
    for tuple_parameter in tuple_parameters:
        nested_keys = tuple_parameter[:-1]
        value = tuple_parameter[-1]
        # Modify the nested dictionary at the given keys,
        # access the nested dictionary directly
        nested_dict = dict_config_new
        for key in nested_keys[:-1]:
            nested_dict = nested_dict[key]
        nested_dict[nested_keys[-1]] = value

    # Create the name of the config file
    seed = dict_config_new["seed"]
    name_env = dict_config_new["env"]["name"]
    max_control = dict_config_new["env"]["params"]["max_control"]
    delay_observation = dict_config_new["env"]["params"]["delay_observation"]
    data_type = dict_config_new["data"]["type"]
    control_type = dict_config_new["data"]["control_type"]

    if name_env in ["cylinder", "pinball", "cavity"]:
        reynolds = dict_config_new["env"]["params"]["reynolds"]
        string_suffix = f"_reynolds_{reynolds}"
    else:
        string_suffix = ""

    name_config_file = "".join(
        f"env_name_{name_env}"
        f"_seed_{seed}"
        f"_max_control_{max_control}"
        f"_delay_observation_{delay_observation}"
        f"_data_type_{data_type}"
        f"_control_type_{control_type}"
        f"{string_suffix}"
        ".yaml"
    )

    print(f"Creating the file {name_config_file}")

    # Create the path of the config file
    path_config_file = f"{path_target_directory}/{name_config_file}"
    # Check if the config file exists
    if pathlib.Path(pathlib.Path(path_config_file).resolve()).exists():
        raise FileExistsError(f"The file {path_config_file} already exists")
    # Write the config file
    with open(path_config_file, "w") as file:
        yaml.dump(dict_config_new, file, default_flow_style=False, sort_keys=False)
