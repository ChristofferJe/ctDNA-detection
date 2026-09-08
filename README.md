# Deep Learning Model for Fragment-Level ctDNA Detection

## Overview

This repository contains the code developed for my research project (master's thesis), *Deep learning model for fragment-level ctDNA detection* which is described in the file `report.pdf`.

It includes implementations of all model variants described in the thesis, together with scripts for training, hyperparameter optimisation, fragment-level and sample-level evaluation, and model interpretation.

## Model Architectures

The following files implement the model variants described in the thesis:

- `CNN/models.py` — convolutional neural network (CNN) variants.
- `RNN/models.py` — recurrent neural network variants, including bidirectional gated recurrent unit (BI-GRU) and bidirectional long short-term memory (BI-LSTM) models.
- `FNN/models.py` — feed-forward neural network (FNN) variants.

Each model variant follows this naming convention:

```text
{architecture}_{fragment_length_preprocessing}_{end_motif_setting}
```

## Data Handling and Utilities

- `cfDataset.py` — implements the PyTorch `Dataset` used to load cfDNA fragments and transform fragmentomic features into the input formats required by each model variant. Fragment data is expected in CSV format.
- `utils.py` — shared functions used throughout the project, including the model training and evaluation pipeline and helpers for data loading and preprocessing.

## Training and Hyperparameter Optimisation

Hyperparameter optimisation is performed with Weights & Biases sweeps.

- `train_with_yaml.py` — main training script. It reads a model and training configuration from a YAML file, then calls `model_pipeline()` in `utils.py` to train and evaluate that configuration.
- `sweep_config_*.yaml` — YAML files defining the search space, optimisation method, and model variant for each architecture. Each configuration calls `train_with_yaml.py` to train and evaluate the configurations in its sweep.
- `train.slurm` — example SLURM job script for running a Weights & Biases hyperparameter-optimisation experiment on an HPC cluster.
- `re_train()` in `utils.py` — continues training from a hyperparameter-search model or starts a new run for one specified configuration.

## Evaluation and Analysis

- `model_evaluation.py` — evaluates trained model variants on the fragment-level test set using ROC-AUC and binary cross-entropy loss.
- `logistic_regression.py` — implements the logistic-regression baseline used to compare against the deep-learning approaches.
- `simulated_sample_level_prediction_stats.py` — performs simulated sample-level predictions for the different architectures.
- `shap_analysis.py` — performs the SHAP (SHapley Additive exPlanations) analysis described in the thesis.

## Notes

- The code was primarily developed and run on an HPC cluster using SLURM.
- Different architectures may require different representations of fragment-end motifs and fragment length; these are handled automatically by the dataset and utility functions.
- Trained model checkpoints and large sequencing datasets are not included in this repository.
- Fragment-end motifs are referred to as `kmers` in some scripts.
