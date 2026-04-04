# Notes on generated plots


## Traing loss

### Toy models
- `ID_XP_DATASET`: 6
- `ID_XP`: 8

#### Files
##### Training loss with respect to $\tau$
- Notebook:
[2024_12_06_visualisation_training_results_phd_thesis.ipynb](2024_12_06_visualisation_training_results_phd_thesis_toy_models.ipynb)
- Images:
[figure_losses_multiple_trajectories.png](data/generated_figures/figure_losses_multiple_trajectories.png)

##### Inference on training data
- Config: [config_toy_models.yaml](configs/archives/config_toy_models.yaml)
- Images: [2024_12_inference_on_training_data_config_2024_12_12_20_27_30.html](data/notebooks_nbconvert/2024_12_inference_on_training_data_config_2024_12_12_20_27_30.html)


#### Remarks
- `.pgf` files are too large...

#### Shell script
```bash
# Generate the inference on training data
./notebook_launcher.sh -n 2024_12_inference_on_training_data.ipynb
```


------------------------------------------------------------


### Navier-Stokes
- `ID_XP_DATASET`: 21
- `ID_XP`: 24

#### Files
##### Training loss with respect to $Re$
- Notebook:
[2024_12_06_visualisation_training_results_phd_thesis.ipynb](2024_12_06_visualisation_training_results_phd_thesis_navier_stokes.ipynb)
- Images:
[figure_losses_cavity.png](data/generated_figures/figure_losses_cavity.png)
[figure_losses_cylinder.png](data/generated_figures/figure_losses_cylinder.png) 
[figure_losses_pinball.png](data/generated_figures/figure_losses_pinball.png)

##### Inference on training data
- Config: [config_navierstokes.yaml](configs/archives/config_navierstokes.yaml)
- Images: [2024_12_inference_on_training_data_config_2024_12_12_20_23_59.html](data/notebooks_nbconvert/2024_12_inference_on_training_data_config_2024_12_12_20_23_59.html)
```bash
# Generate the inference on training data
# With config_navierstokes.yaml
./notebook_launcher.sh -n 2024_12_inference_on_training_data.ipynb
```


------------------------------------------------------------


### Mackey-Glass
- `ID_XP_DATASET`: 18
- `ID_XP`: 27

#### Files
##### Training loss with respect to $\tau$

##### Inference on training data
- Config:
[config_mackey_glass.yaml](configs/archives/config_mackey_glass.yaml)
- Images:
[2024_12_inference_on_training_data_config_2024_12_11_18_00_18.html](data/notebooks_nbconvert/2024_12_inference_on_training_data_config_2024_12_11_18_00_18.html)


#### Shell script
```bash
# Generate the inference on training data
# With config_mackey_glass.yaml
./notebook_launcher.sh -n 2024_12_inference_on_training_data.ipynb
```