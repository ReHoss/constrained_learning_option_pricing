# Project instructions

- The core of the package is in `learning_option_pricing/` while the python scripts used for experiments are under `experiments/`.
- The generated data from experiments should be stored in `data/<name_of_python_script>/<xp_folder>` where `<xp_folder>` is a timestamped folder containing the principal config values for the experiment.
- There is already some code under `experiments/` coming from a SciML project; this may be used as a reference for how to structure the experiments and store results.
- Please update `pyproject.toml` when needed and also the contribution guidelines in `CONTRIBUTING.md`.
- Document the implementation in some manual under documents/methodology/ !
- In markdown files, please use LaTeX for mathematical expressions.
- Log as much information as possible in experiments and scripts.
