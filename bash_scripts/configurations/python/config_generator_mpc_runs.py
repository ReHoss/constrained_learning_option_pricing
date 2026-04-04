# Create a generator of config files for python
import argparse
import datetime
import itertools
import logging

# import itertools
import pathlib

import mlflow
import yaml
from control_dde.utils import PATH_PROJECT_ROOT
from mlflow import entities
from test.mpc.test_data_generator_and_training import (
    get_yaml_config_from_folder,
    make_dict_env,
    make_dict_mpc_env,
)


# TODO: Check the config file quality


def get_list_path_experiment(
    path_mlflow_uri: pathlib.Path, id_mlflow_experiment: int
) -> list[pathlib.Path]:
    # Warning: @RH: I use MLFlow 1.27.0
    # Set the MLFlow tracking URI
    logging.info(
        "Experiment wil be retrieved from %s",
        path_mlflow_uri,
    )
    mlflow.set_tracking_uri(f"file://{path_mlflow_uri}")  # TODO: May be fix
    # Retrieve the experiment path
    experiment: entities.Experiment | None
    experiment = mlflow.get_experiment(str(id_mlflow_experiment))

    if experiment is None:
        raise ValueError(f"Experiment {id_mlflow_experiment} not found.")

    # Get the path of the experiment
    # path_xp_traj_generator = experiment.artifact_location.replace("file://", "")
    # Work on the path to make it platform independent, replace its project root
    # by the project root from the current script
    # Extract the experiment path from ./data/mlruns
    # relative_path_xp_traj_generator = path_xp_traj_generator.removeprefix(
    #     f"{PATH_PROJECT_ROOT}/"
    # )
    path_xp_traj_generator = pathlib.Path(
        f"{path_mlflow_uri}/{experiment.experiment_id}"
    )

    path_xp_traj_generator = pathlib.Path(path_xp_traj_generator)

    # Loop through the directories of the experiment
    # and retrieve the trajectory data
    # Pyright does not like list(path_xp_traj_generator.iterdir())

    list_dir_to_exclude = ["tags"]

    return [
        path
        for path in path_xp_traj_generator.iterdir()
        if path.is_dir() and path.name not in list_dir_to_exclude
    ]


def _check_spurious_combinations(tuple_parameters: tuple) -> bool:
    _ = tuple_parameters
    # Return True if the combination is spurious

    # tuple_name_model = tuple_parameters[1]
    # tuple_name_trainer = tuple_parameters[2]
    # tuple_n_delays = tuple_parameters[3]
    # # tuple_path_data_folder = tuple_parameters[4]
    #
    # if tuple_name_model[-1] != tuple_name_trainer[-1]:
    #     return True
    #
    # if tuple_name_model[-1] in ["node", "ncde"]:
    #     if tuple_n_delays[-1] != 0:
    #         return True
    # if tuple_name_model[-1] in ["ndde", "ncdde"]:
    #     if tuple_n_delays[-1] == 0:
    #         return True

    # TODO: Check assert dict_model["name"] == dict_trainer["name"]

    return False


"""
1 - Do not forget to change the mlfow_experiment_name value
2 - The file naming may impact file generation when overwriting is permitted, an
    exception has been included to prevent this
"""


# noinspection PyUnusedLocal
def _post_process_config_file(dict_config: dict) -> None:
    _ = dict_config
    # Post-process the config file

    # Remove the n_delays key if the model is not a DDE

    # if dict_config["model"]["name"] not in ["ndde", "ncdde"]:
    #     dict_config["model"]["params"].pop("n_delays", None)


def get_yaml_config_from_training_run_path(
    path_experiment_training: pathlib.Path,
) -> dict:
    # Load the configuration from the training
    # Open and load the yaml file
    path_config_file = pathlib.Path(f"{path_experiment_training}/artifacts/config")
    dict_config_training = get_yaml_config_from_folder(path_config_file)
    return dict_config_training


def get_data_generator_config_from_path_data_folder(
    path_data_folder: pathlib.Path,
) -> dict:
    path_data_generator_config = pathlib.Path(
        f"{str(path_data_folder).removesuffix('generated_data/trajectory_data')}"
        f"/artifacts/config"
    )
    dict_data_generator = get_yaml_config_from_folder(path_data_generator_config)
    return dict_data_generator


# noinspection PyUnusedLocal
def generate_mpc_config_from_training_config(
    # dict_config_mpc: dict, dict_config_training: dict, dict_data_generator: dict
) -> dict:
    return {}


def update_dict_config_base_mpc_from_training_config(
    dict_base_mpc_config: dict,
    mlflow_experiment_name: str,
    dict_env: dict,
    dict_mpc_env: dict,
) -> dict:
    # Copy dictionary
    dict_base_mpc_config_copy = dict_base_mpc_config.copy()
    dict_env_copy = dict_env.copy()
    dict_mpc_env_copy = dict_mpc_env.copy()

    # Update the dictionary
    dict_base_mpc_config_copy["mlflow_experiment_name"] = mlflow_experiment_name
    dict_base_mpc_config_copy["env"] = dict_env_copy
    dict_base_mpc_config_copy["mpc_env"] = dict_mpc_env_copy

    return dict_base_mpc_config_copy


def generate_config_mpc_from_dict_config_training(
    dict_config_training: dict,
    dict_config_mpc_base: dict,
    name_mlflow_experiment_mpc: str,
    path_trained_model: pathlib.Path,
) -> dict:
    dict_model = dict_config_training["model"]
    dict_trainer = dict_config_training["trainer"]
    path_data_folder = dict_config_training["data"]["path_data_folder"]

    # Get the config of the trajectory generator
    path_data_generator_config = pathlib.Path(
        f"{str(path_data_folder).removesuffix('generated_data/trajectory_data')}"
        f"/artifacts/config"
    )
    dict_data_generator = get_yaml_config_from_folder(
        PATH_PROJECT_ROOT / path_data_generator_config
    )

    # Generate the dict for the ground truth environment
    dict_env = make_dict_env(
        dict_data_generator=dict_data_generator
    )  # TODO: Warning delay is popped out

    dict_mpc_env_params = {}
    dict_mpc_env = make_dict_mpc_env(
        dict_model=dict_model,
        dict_trainer=dict_trainer,
        path_trained_model=path_trained_model,
        dict_mpc_env_params=dict_mpc_env_params,
    )

    dict_config_mpc = update_dict_config_base_mpc_from_training_config(
        dict_base_mpc_config=dict_config_mpc_base,
        mlflow_experiment_name=name_mlflow_experiment_mpc,
        dict_env=dict_env,
        dict_mpc_env=dict_mpc_env,
    )
    return dict_config_mpc


def generate_config_mpc_from_path_run(
    path_experiment_training: pathlib.Path,
    dict_config_mpc_base: dict,
    name_mlflow_experiment_mpc: str,
) -> dict:
    # Load the configuration from the training
    # Open and load the yaml file  # TODO: Set this!
    path_config_file = pathlib.Path(f"{path_experiment_training}/artifacts/config")
    dict_config_training = get_yaml_config_from_folder(path_config_file)

    path_trained_model = pathlib.Path(
        f"{path_experiment_training}/generated_data/"
        f"trainer_data/model_data".removeprefix(f"{PATH_PROJECT_ROOT}/")
    )

    return generate_config_mpc_from_dict_config_training(
        dict_config_training=dict_config_training,
        dict_config_mpc_base=dict_config_mpc_base,
        name_mlflow_experiment_mpc=name_mlflow_experiment_mpc,
        path_trained_model=path_trained_model,
    )


def change_dict_keys(dict_config: dict, tuple_parameters: tuple) -> dict:
    # noinspection DuplicatedCode
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
    return dict_config_new


# TODO: Automatise the xp name generation with time or random symbol
# noinspection PyPep8Naming
def main():
    text_yaml_config_file = """
    # ----- MLFlow Experiment -----
    mlflow_experiment_name: debug2
    device: cpu
    dtype: float32
    seed: 94
    
    append_data: false
    append_data_eval: false
    
    # Checkpoints
    checkpoints:
      load: false
      restart_every_n_iter: null
      save: true
      save_every_n_iter: 1
    
    # Controller type and parameters
    controller: mpc-icem
    controller_params:
      action_sampler_params:
        alpha: 0.1
        elites_size: 10
        fraction_elites_reused: 0.3
        init_std: 0.5
        keep_previous_elites: true
        noise_beta: 2.5
        opt_iterations: 3
        shift_elites_over_time: true
        use_mean_actions: true
      cost_along_trajectory: sum
      do_visualize_plan: true
      factor_decrease_num: 1.25
      horizon: 20
      num_simulated_trajectories: 5  # TODO RESET
      verbose: true
      use_env_reward_as_cost: true
    
    # Ground truth environment
    env:
      name: pendulum
      params: {}
    
    # Model used for MPC
    mpc_env:
      name: ncde
      path_model: >
        data/examples/python_scripts/training_on_trajectory_data/
        node_max_epochs-5000/generated_data/trainer_data/model_data
      params: {}
      model:
        name: ncde
        params:
          width_size: 64
          depth: 2
          dropout: 0
          activation: elu
          final_activation: identity
      trainer:
        name: ncde
        params:
          name_node_interface: torchdde
          solver: euler
          lr_init: 0.0005
          lr_final: 0.0005
          max_epochs: 5000
          bool_discretize_then_optimize: false
          rate_test_set_size: 0.1
          pct_trajectory_size: 0.25
    
    
    evaluation_rollouts: 0
    
    # Forward model
    #forward_model: ParallelGroundTruthModel  # This allows to parallelise computation
    forward_model: GroundTruthModel  # This allows to parallelise computation
    forward_model_params: {}
    #  num_parallel: 8
    
    # Initial controller (@ReHoss: I don't know yet what this is, but set it to none)
    initial_controller: none
    initial_controller_params: { }
    initial_number_of_rollouts: 0
    
    # Rollouts (number of independant runs of MPC control
    number_of_rollouts: 1
    rollout_params:
      only_final_reward: false
      record: false
      render: false
      render_eval: false
      render_initial: false
      task_horizon: 100  # TODO RESET
      use_env_states: true   # Always set to true otherwise it triggers state from obs
    
    # Training (an interface is provided to train and improve the forward model)
    training_iterations: 1
    """

    # mlflow_experiment_name = (f"{dict_config_training["mlflow_experiment_name"]}"
    #                           f"_mpc_xp_{datetime.datetime.now()
    #                           .strftime('%Hh_%Mm')}")

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

    # noinspection DuplicatedCode
    path_target_directory = pathlib.Path(
        f"{PATH_PROJECT_ROOT}/config/batch/{name_target_directory}"
    ).resolve()
    # Check if the target directory exists
    if not pathlib.Path(pathlib.Path(path_target_directory).resolve()).exists():
        # Create the target directory
        pathlib.Path(path_target_directory).mkdir(parents=False, exist_ok=False)

    # Load the yaml text as a dictionary
    dict_config_mpc_base = yaml.load(text_yaml_config_file, Loader=yaml.FullLoader)

    # Add a suffix to the mlflow_experiment_name to differentiate the runs
    current_time = datetime.datetime.now().strftime("%Hh_%Mm")

    # --- Start of the list of parameters to change
    # Define the list of parameters to change,
    # by giving in the tuple the nested keys of the dictionary

    # ----
    # ID of the MLFlow experiment for the training
    ID_MLFLOW_XP_TRAINING = 8
    # Path to the MLFlow runs - this approach needs to be platform independent
    MLFLOW_STORE_NAME = "mlruns"
    PATH_MLFLOW_URI = pathlib.Path(f"{PATH_PROJECT_ROOT}/data/{MLFLOW_STORE_NAME}")

    # Needed because all paths are relative to the project root
    # PATH_CONTENT_ROOT_TO_REMOVE = pathlib.Path(
    #     "/home/hosseinkhan/Documents/work/phd/git_repositories/control_dde/"
    # )  # TODO: Fix this by checking the paths...

    # Define the name of the MLFlow experiment for the MPC
    # First, get the experiment name
    mlflow.set_tracking_uri(f"file://{PATH_MLFLOW_URI}")

    name_training_experiment = mlflow.get_experiment(str(ID_MLFLOW_XP_TRAINING)).name
    name_mlflow_experiment_mpc = f"{name_training_experiment}_mpc_xp_{current_time}"

    list_path_experiment: list[pathlib.Path] = get_list_path_experiment(
        path_mlflow_uri=PATH_MLFLOW_URI,
        id_mlflow_experiment=ID_MLFLOW_XP_TRAINING,
    )

    # path_run = list_path_experiment[0]

    # ----

    N_SEEDS = 2
    list_seed = [("seed", seed) for seed in range(N_SEEDS)]

    # TODO: Don't forget to skip spurious cases

    # !!!! UPDATE THE NESTED LIST OF PARAMETERS TO CHANGE HERE !!!!
    nested_list_parameters = [
        list_seed,
    ]

    # Loop over runs and for each of them, create the config files
    for path_run in list_path_experiment:
        dict_config_run_generated_config_mpc = generate_config_mpc_from_path_run(
            path_experiment_training=path_run,
            dict_config_mpc_base=dict_config_mpc_base,
            name_mlflow_experiment_mpc=name_mlflow_experiment_mpc,
        )
        write_yaml_config_with_parameters_change(
            dict_config_mpc=dict_config_run_generated_config_mpc,
            nested_list_parameters=nested_list_parameters,
            path_run=path_run,
            path_target_directory=path_target_directory,
        )


def write_yaml_config_with_parameters_change(
    dict_config_mpc: dict,
    nested_list_parameters: list[list[tuple]],
    path_run: pathlib.Path,
    path_target_directory: pathlib.Path,
):
    for tuple_parameters in itertools.product(*nested_list_parameters):
        # Check for spurious combinations
        if _check_spurious_combinations(tuple_parameters):
            continue

        # Create a new dictionary with the new parameters
        dict_config_new = change_dict_keys(
            dict_config=dict_config_mpc, tuple_parameters=tuple_parameters
        )

        # Post-process the config file
        _post_process_config_file(dict_config=dict_config_new)

        seed = dict_config_new["seed"]
        run_id = path_run.stem

        name_config_file = "".join(f"run_id_{run_id}" f"_seed_{seed}" ".yaml")

        print(f"Creating the file {name_config_file}")

        # Create the path of the config file
        # noinspection DuplicatedCode
        path_config_file = f"{str(path_target_directory)}/{name_config_file}"
        # Check if the config file exists
        if pathlib.Path(pathlib.Path(path_config_file).resolve()).exists():
            raise FileExistsError(f"The file {path_config_file} already exists")
        # Write the config file
        with open(path_config_file, "w") as file:
            yaml.dump(dict_config_new, file, default_flow_style=False, sort_keys=False)


# --- End of the list of parameters to change

# """
if __name__ == "__main__":
    main()
