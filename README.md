# Cooperation Task Model Fitting

## Description

This repository contains MATLAB scripts for fitting computational models to cooperation task behavioral data. 
There are three folders:
The AInf_scripts folder contains the most up-to-date Active Inference scripts.
The AInf_scripts_used_to_compare_VB_with_MCMC folder contains slightly out-of-date active inference scripts to compare fitting the same model with Variational Bayes (here) vs. Markov Chain Monte Carlo Sampling (https://github.com/cgoldman123/Active_Inference_Model_MCMC_Cooperation_Task)
The RL_heuristic_scripts contains Reinforcement Learning scripts.

Each of these pipelines implement model fitting using **variational Bayes**, simulate fitted parameters, and generate **model-free analyses**. The code can operate in two experimental modes: **local** (for in-house experiments) and **prolific** (for online data collection). The results are saved in `.csv` and `.mat` formats for further analysis.

---

## Scripts used in each pipeline:

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

### Model scripts: `Simple_TAB_model_v3.m`, `RW_model.m`, `Simple_TAB_model_v2.m`, etc

- **Purpose**: Implements the computational models for simulating and fitting cooperation task data.
- **Functionality**:
  - Computes action probabilities by modeling subjects' learning and decision-making.
  - Supports both **simulation** and **fitting** modes.
  - Incorporates **learning rates** and **forgetting rates** that can be split based on outcome type (wins, losses, neutral).
  - Outputs key model variables like chosen actions, outcomes, action probabilities, and model parameters.





