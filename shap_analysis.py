import shap
import pandas as pd
from cfDataset import cfDataset
import numpy as np
from CNN.models import *
from RNN.models import *
from FNN.models import *
from utils import WrappedModel, dataset_to_tensors
import wandb
import torch

# alias pythoncf='LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python'

# Get background and explainer samples
background_samples = pd.read_csv("../data/shap/background_samples.csv")
explainer_samples = pd.read_csv("../data/shap/explained_samples.csv")

# Testing sampels
#background_samples = background_samples.sample(n=100, random_state=2)
#explainer_samples = explainer_samples.sample(n=5000, random_state=2)

# Get mean and sd of training length for normalization
train_length_mean, train_length_std = np.load("normalization_stats/length_stats_kmer_combined.npy")

# Make dataset for the background and explainer samples
background_dataset = cfDataset(background_samples, mean = train_length_mean, std = train_length_std, normalize=True, kmer_combined=True)
explainer_dataset = cfDataset(explainer_samples, mean = train_length_mean, std = train_length_std, normalize=True, kmer_combined=True)

# Transform the dataset into tensors as required for the shapley explainer
background_start_kmers, background_end_kmers, background_lengths, background_labels = dataset_to_tensors(background_dataset)
explainer_start_kmers, explainer_end_kmers, explainer_lengths, explainer_labels = dataset_to_tensors(explainer_dataset)

design = "GRU"
print(f"Running SHAP analysis for {design} model", flush=True)

# Load the trained model
api = wandb.Api()

if design == "CNN":
    run = api.run("christoffer-je-aarhus-university/cfDNA/3pvqujqd") # from sweep with configs (chosen rank)
    model = CNN_normalize_combined_end_motif(num_filters=run.config['num_filters'],
                                        kernel_size=run.config['kernel_size'],
                                        length_hidden_size=run.config['length_hidden_size'],
                                        fusion_hidden_size=run.config['fusion_hidden_size'])
    artifact = api.artifact("christoffer-je-aarhus-university/cfDNA/model:v68246") # Best config trained on train and val

if design == "LSTM":
    run = api.run("christoffer-je-aarhus-university/cfDNA/4bv35gd2") # from sweep with configs (chosen rank)
    model = LSTM_BI_normalize_combined_end_motif(lstm_hidden_size=run.config['lstm_hidden_size'], 
                            length_hidden_size=run.config['length_hidden_size'],
                            fusion_hidden_size=run.config['fusion_hidden_size'])
    artifact = api.artifact("christoffer-je-aarhus-university/cfDNA/model:v40206") # Best config trained on train and val

if design == "GRU":
    run = api.run("christoffer-je-aarhus-university/cfDNA/lxt8g9y1") # from sweep with configs (chosen rank)
    model = GRU_BI_normalize_combined_end_motif(gru_hidden_size=run.config['gru_hidden_size'], 
                            length_hidden_size=run.config['length_hidden_size'],
                            fusion_hidden_size=run.config['fusion_hidden_size'])
    artifact = api.artifact("christoffer-je-aarhus-university/cfDNA/model:v40207") # Best config trained on train and val

if design == "FNN":
    run = api.run("christoffer-je-aarhus-university/cfDNA/102p1irm") # from sweep with configs (chosen rank)
    model = FNN_normalize_combined_end_motif(end_motif_hidden_size=run.config['end_motif_hidden_size'],
                                  length_hidden_size=run.config['length_hidden_size'],
                                  fusion_hidden_size=run.config['fusion_hidden_size'])
    artifact = api.artifact("christoffer-je-aarhus-university/cfDNA/model:v68245") # Best config trained on train and val

artifact_dir = artifact.download()
model.load_state_dict(torch.load(f'{artifact_dir}/model.pth'))
model.eval()

# Wrap model in shap wrapper
wrapped_model = WrappedModel(model)
wrapped_model.eval()

# Build the explainer using background samples
explainer = shap.GradientExplainer(
    wrapped_model,
    [
        background_start_kmers,
        background_end_kmers,
        background_lengths
    ]
)

# Make shap analysis on the explainer samples
shap_values = explainer.shap_values(
    [
        explainer_start_kmers,
        explainer_end_kmers,
        explainer_lengths
    ]
)

# Remove extra unecessary dimension from shap values
start_shap = np.squeeze(shap_values[0], axis=-1)
end_shap = np.squeeze(shap_values[1], axis=-1)
length_shap = shap_values[2]

# Shap results
shap_results = {
    "start_shap": start_shap,
    "end_shap": end_shap,
    "length_shap": length_shap
}

# Combine values for start and end kmers
combined_shap = np.concatenate([np.abs(start_shap), np.abs(end_shap)], axis=0)

# Get importance per nucleotide position
position_importance = combined_shap.mean(axis=(0,2))

# Get importance per nuclotide position per nuclotide value
position_value_importance = combined_shap.mean(axis=(0))

# Get overall fragment length importance
length_importance = np.abs(length_shap).mean()

# Save all the results 
np.savez_compressed(
    f"../data/shap/results/{design}_shap_results.npz",
    start_shap=start_shap,
    end_shap=end_shap,
    length_shap=length_shap
)

np.save(
    f"../data/shap/results/{design}_position_importance.npy", 
    position_importance)

np.save(
    f"../data/shap/results/{design}_position_value_importance.npy", 
    position_value_importance)

np.save(
    f"../data/shap/results/{design}_length_importance.npy",
    length_importance
)