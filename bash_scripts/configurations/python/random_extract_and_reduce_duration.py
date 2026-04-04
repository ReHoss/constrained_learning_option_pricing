import os
import random
import re


# Specify the folder containing .yaml files
folder_path = (
    "/gpfs/workdir/hosseinkhanr/pycharm_remote_project/"
    "control_dde/config/batch/test_debug"
)

# Number of files to keep
num_files_to_keep = 15  # Change to 15 if needed

# Get all .yaml files in the folder
all_files = [f for f in os.listdir(folder_path) if f.endswith(".yaml")]

# Randomly select files to keep
files_to_keep = random.sample(all_files, num_files_to_keep)
print(f"Files to keep: {files_to_keep}")

# Remove files not in the list
for filename in all_files:
    if filename not in files_to_keep:
        os.remove(os.path.join(folder_path, filename))
        print(f"Deleted: {filename}")

# Replace the parameter in the remaining files
for filename in files_to_keep:
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r") as file:
        content = file.read()

    # Replace task_horizon: 100 with task_horizon: 6
    updated_content = re.sub(r"task_horizon:\s*100", "task_horizon: 6", content)

    with open(file_path, "w") as file:
        file.write(updated_content)
        print(f"Updated: {filename}")
