# Cooperation Task Model Fitting

## Description

This repository contains MATLAB scripts for fitting computational models to cooperation task behavioral data. The main script and associated functions implement model fitting using **variational Bayes**, simulate fitted parameters, and generate **model-free analyses**. The code can operate in two experimental modes: **local** (for in-house experiments) and **prolific** (for online data collection). The results are saved in `.csv` and `.mat` formats for further analysis.

---

## Main Scripts and Functions

### `main_script.m`

- **Purpose**: The primary script for fitting models to cooperation task data.
- **Key Features**:
  - Fits models using variational Bayes for each subject listed in `fit_list`.
  - Supports parameter simulation using pre-defined values (`SIM_PASSED_PARAMETERS`).
  - Saves model-free analysis results if `DO_MODEL_FREE` is set to `true`.
  - Allows for saving prediction errors and fitting results in the specified `result_dir`.

### `TAB_fit_simple_prolific.m`

- **Purpose**: Fits cooperation task data for Prolific subjects.
- **Workflow**:
  - Reads behavioral data files.
  - Parses observations and actions.
  - Sets up the MDP (Markov Decision Process) model structure.
  - Inverts the model using variational Bayes via `TAB_inversion_simple`.

### `TAB_inversion_simple.m`

- **Purpose**: Inverts the MDP model using variational Bayes.
- **Inputs**:
  - `DCM`: Contains the MDP structure, parameters to optimize, outcomes (`DCM.U`), and responses (`DCM.Y`).
- **Outputs**:
  - `DCM.M`: The generative model.
  - `DCM.Ep`: Posterior means of parameters.
  - `DCM.Cp`: Posterior covariances.
  - `DCM.F`: Variational free energy (used as a log evidence bound).

### `Simple_TAB_model_v2.m`

- **Purpose**: Implements the Simple TAB (Tree-based Active Inference) model for simulating and fitting cooperation task data.
- **Functionality**:
  - Computes action probabilities, expected free energies (EFE), and updates reward probabilities.
  - Supports both **simulation** and **fitting** modes.
  - Incorporates **learning rates** and **forgetting rates** that can be split based on outcome type (wins, losses, neutral).
  - Outputs key model variables like chosen actions, outcomes, action probabilities, and model parameters.

#### Output Fields

- **`model_output.choices`**: Selected actions.
- **`model_output.outcomes`**: Observed outcomes.
- **`model_output.action_probabilities`**: Probabilities for each possible action.
- **`model_output.chosen_action_probabilities`**: Probabilities of the chosen actions.
- **`model_output.learned_reward_probabilities`**: Updated reward probabilities.
- **`model_output.EFE`**: Expected free energies.
- **`model_output.G_error`**: Prediction error based on surprise.

---

## Key Parameters and Variables

- **Experimental Modes**:
  - `local`: For in-house experiments.
  - `prolific`: For online experiments via Prolific.

- **Simulation Parameters**:
  - `simmed_alpha`, `simmed_cr`, `simmed_cl`, `simmed_eta`, `simmed_omega`, `simmed_opt`: Pre-defined parameter values for simulations.

- **Model Settings**:
  - `DCM.MDP.forgetting_split_matrix` / `DCM.MDP.forgetting_split_row`: Controls forgetting rate split (wins/losses/neutral).
  - `DCM.MDP.learning_split`: Toggles learning rate split (wins/losses/neutral).

---

## Usage

1. **Set Experimental Mode**: Modify `experiment_mode` in `main_script.m` (`"local"` or `"prolific"`).
2. **Specify Subjects**: Update `fit_list` with appropriate subject IDs.
3. **Run the Main Script**:
   ```matlab
   main_script
