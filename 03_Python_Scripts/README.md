# Python Automation & Processing Scripts

This folder contains all the Python scripts developed for the thesis. Detailed explanations of the methodology and data flow are included as docstrings inside each individual script. 

The scripts are categorized below according to their role in the research pipeline:

### 1. Abaqus Automation & Data Extraction
* **`cruciform_automation.py`**: Automates finite element simulations and data extraction (forces and strains) for the cruciform test using the CPB'06 yield criterion and Swift law.
* **`shear_automation.py`**: Automates simple shear test simulations by dynamically updating a master CAE model and extracting boundary forces and center strains.
* **`asymmetric_automation.py`**: Automates the asymmetric notched tensile test, featuring a batch control mechanism to process large DoE datasets.

### 2. Baseline Machine Learning (XGBoost)
* **`xgboost_train.py`**: Implements the baseline training pipeline using PCA dimensionality reduction and MultiOutput XGBoost regression.
* **`xgboost_testing.py`**: Evaluates the generalization capability of the trained XGBoost model on unseen testing data.

### 3. Artificial Neural Networks (ANN)
* **`NN_model_optimization.py`**: Utilizes Bayesian optimization (Optuna) to dynamically identify the optimal hyperparameter topology for the ANN.
* **`training_NN_model.py`**: Executes the final training, validation, and early-stopping pipeline for the optimized ANN architecture.
* **`testing_and_plot_NN_model.py`**: Reconstructs the exact data splits to test the model and generates publication-ready parity plots with gradient coloring.

### 4. Advanced & Physics-Informed Methodologies
* **`train_test_plot_PINN_model.py`**: Enhances the baseline ANN by introducing a physics-guided composite loss function (computing real-time yield and hardening) and merging shear/cruciform data.
* **`train_test_plot_Q_effn.py`**: Predicts CPB'06 parameters within its canonical space (Q_eff,n) and features a custom decoding algorithm to rebuild the original anisotropic matrix for a new DoE.
* **`train_test_plot_Ccan.py`**: Implements the final formulation strategy, directly targeting the variables within the canonical parameter matrix (C_can).

### 5. Macroscopic Validation
* **`analytical_numerical_validation.py`**: Performs the final validation by cross-referencing ground-truth FE simulations with the pipeline's predictions, isolating median and worst-case scenarios to prove macroscopic and physical fidelity.
