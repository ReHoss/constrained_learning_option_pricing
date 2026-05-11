# Project instructions

- The core of the package is in `learning_option_pricing/` while the python scripts used for experiments are under `experiments/`.
- The generated data from experiments should be stored in `data/<name_of_python_script>/<xp_folder>` where `<xp_folder>` is a timestamped folder containing the principal config values for the experiment.
- There is already some code under `experiments/` coming from a SciML project; this may be used as a reference for how to structure the experiments and store results.
- Please update `pyproject.toml` when needed and also the contribution guidelines in `CONTRIBUTING.md`.
- Document the implementation by updating the markdown files under `documents/methodology/`.
- In markdown files, please use LaTeX for mathematical expressions.
- Log as much information as possible in experiments and scripts.
- the virtual environment is installed here venv/venv_learning_option_pricing
- Save the run statistics so that plots can be easily updated without re-running the experiments.
- Please also make sure the metrics are (reasonably) recorded per run so that figure patches can be applied without re-running the experiments
- Add a textbox below the figures that give the analytical formula for the symbols used in the figures.
- Labels figures when you think comparison are unfair.
- When running experiments, give the command to follow the progress of the experiment in real time. Please give the full absolute path to the log file so that it can be easily followed.
- be verbose, minimise acronyms, except for well-known ones. Be verbose in general with file and folder names, variable names, and comments in the code. This will make it easier for others to understand the code and contribute to the project.
- Code documentation must be in English.
- Favour muliple small commits with descriptive commit messages over large commits with vague messages.
