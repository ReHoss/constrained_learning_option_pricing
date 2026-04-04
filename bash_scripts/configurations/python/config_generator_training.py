# Create a generator of config files for python
import argparse
import datetime
import itertools
import logging
import pathlib

import mlflow
import yaml
from mlflow import entities


def list_path_experiment(
    path_mlflow_uri: pathlib.Path, mlflow_experiment_name: str
) -> list[pathlib.Path]:
    # Warning: @RH: I use MLFlow 1.27.0
    # Set the MLFlow tracking URI
    logging.debug(
        "Experiment wil be retrieved from %s",
        path_mlflow_uri,
    )
    mlflow.set_tracking_uri(f"file://{path_mlflow_uri}")  # TODO: May be fix
    # Retrieve the experiment path
    experiment: entities.Experiment | None
    experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {mlflow_experiment_name} not found.")

    path_xp_traj_generator = experiment.artifact_location.replace("file://", "")
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
    # Return True if the combination is spurious

    tuple_name_model = tuple_parameters[1]
    tuple_name_trainer = tuple_parameters[2]
    tuple_n_delays = tuple_parameters[3]
    # tuple_path_data_folder = tuple_parameters[4]

    if tuple_name_model[-1] != tuple_name_trainer[-1]:
        return True

    if tuple_name_model[-1] in ["node", "ncde"]:
        if tuple_n_delays[-1] != 0:
            return True
    if tuple_name_model[-1] in ["ndde", "ncdde"]:
        if tuple_n_delays[-1] == 0:
            return True

    return False


"""
1 - Do not forget to change the mlfow_experiment_name value
2 - The file naming may impact file generation when overwriting is permitted, an
    exception has been included to prevent this
"""


def _post_process_config_file(dict_config: dict) -> None:
    # Post-process the config file
    # Remove the n_delays key if the model is not a DDE
    if dict_config["model"]["name"] not in ["ndde", "ncdde"]:
        dict_config["model"]["params"].pop("n_delays", None)


# TODO: Automatise the xp name generation with time or random symbol
# noinspection PyPep8Naming
def main():
    text_yaml_config_file = f"""
    seed: 0
    mlflow_experiment_name: {datetime.datetime.now().strftime('%Y_%m_%d')}_training
    
    model:
      name: ncde
      params:
        width_size: 64
        depth: 2
        dropout: 0
        activation: relu
        final_activation: identity
        n_delays: 1
        delay_exponential_dist_param: 1.0
    
    trainer:
      name: ncde
      params:
        name_node_interface: torchdde
        solver: rk4
        lr_init: 0.0001
        lr_final: 0.0001
        max_epochs: 3000
        bool_discretize_then_optimize: true
        rate_test_set_size: 0.1
        pct_trajectory_size: 1.0
    
    
    data:
      path_data_folder: data/examples/python_scripts/generating_trajectory_data/.....
      normaliser:
        name: standard_normal
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

    # ----
    MLFLOW_XP_NAME_DATASET = "2024_09_25_training_data_14h_50m"
    PATH_MLFLOW_URI = "/home/firedrake/mount_dir/project_root/data/mlruns"
    PATH_ROOT_TO_REMOVE = "/home/firedrake/mount_dir/project_root/"
    # MLFLOW_XP_NAME_DATASET = "debug_pytest_2024-08-17_00-54-25_trajectory_generator"
    # PATH_MLFLOW_URI = "/tmp/pytest-of-hosseinkhan/pytest-10/mlruns"
    # PATH_ROOT_TO_REMOVE = "/tmp/"

    LIST_MODEL_NAME = ["node", "ncde", "ndde", "ncdde"]
    # ----

    n_seeds = 2
    list_seed = [("seed", seed) for seed in range(n_seeds)]

    list_name_model = [("model", "name", name_model) for name_model in LIST_MODEL_NAME]
    list_name_trainer = [
        ("trainer", "name", name_trainer) for name_trainer in LIST_MODEL_NAME
    ]

    n_delays = 1
    list_n_delays = [("model", "params", "n_delays", n) for n in range(0, n_delays + 1)]

    list_path_data_folder = [
        (
            "data",
            "path_data_folder",
            f"{str(path_data_folder).removeprefix(PATH_ROOT_TO_REMOVE)}"
            f"/generated_data/trajectory_data",
        )
        for path_data_folder in list_path_experiment(
            path_mlflow_uri=pathlib.Path(PATH_MLFLOW_URI),
            mlflow_experiment_name=MLFLOW_XP_NAME_DATASET,
        )
    ]

    # TODO: Don't forget to skip spurious cases

    # !!!! UPDATE THE NESTED LIST OF PARAMETERS TO CHANGE HERE !!!!
    nested_list_parameters = [
        list_seed,
        list_name_model,
        list_name_trainer,
        list_n_delays,
        list_path_data_folder,
    ]

    for tuple_parameters in itertools.product(*nested_list_parameters):
        # Check for spurious combinations
        if _check_spurious_combinations(tuple_parameters):
            continue

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

        _post_process_config_file(dict_config=dict_config_new)

        id_dataset = pathlib.Path(
            tuple_parameters[-1][-1].removesuffix("/generated_data/trajectory_data")
        ).stem
        # Create the name of the config file
        name_config_file = "".join(
            f"{id_dataset}_{tuple_parameter[-2]}_{tuple_parameter[-1]}_"
            for tuple_parameter in tuple_parameters
            if tuple_parameter[0] != "data"
        )
        name_config_file = f"{name_config_file[:-1]}.yaml"
        # Create the path of the config file
        path_config_file = f"{str(path_target_directory)}/{name_config_file}"
        # Check if the config file exists
        if pathlib.Path(pathlib.Path(path_config_file).resolve()).exists():
            raise FileExistsError(f"The file {path_config_file} already exists")
        # Write the config file
        with open(path_config_file, "w") as file:
            logging.debug("Writing the config file %s", path_config_file)
            print(f"Writing the config file {path_config_file}")
            yaml.dump(dict_config_new, file, default_flow_style=False, sort_keys=False)


# --- End of the list of parameters to change

if __name__ == "__main__":
    main()
