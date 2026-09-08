import pandas as pd
from cfDataset import cfDataset
import numpy as np
from CNN.models import *
from RNN.models import *
from FNN.models import *
import wandb
from torch.utils.data import DataLoader
import torch
import sys
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from utils import WrappedLogRegModel
from joblib import load
import seaborn as sns
from sklearn.metrics import log_loss

#alias pythoncf='LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python'


# Print the fragment level test AUC and loss for each design
'''
designs = ["GRU", "LSTM", "CNN", "FNN","LogReg"]
titles = ["BI-GRU", "BI-LSTM", "CNN", "FNN", "Logistic Regression"]

# Get fragment labels
test_df = pd.read_csv("../data/test_from10.csv")
y_true = test_df['class'].values

# Define the loss
loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
loss_for_reg = torch.nn.BCELoss(reduction="mean")

for design, title in zip(designs, titles):
    predictions = np.load(f"../data/prediction_stats/best_{design}_test_fragment_level_predictions.npy")

    fpr, tpr, _ = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
        
    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc_auc:.4f})")

    # Calcuate and print the loss for the design
    if design != "LogReg":
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
        design_loss = loss(predictions_tensor, y_true_tensor).item()
        print(f"{title} Loss: {design_loss:.4f}")
        print(f"{title} AUC: {roc_auc:.4f}")
    else:
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
        design_loss = loss_for_reg(predictions_tensor, y_true_tensor).item()
        print(f"{title} Loss: {design_loss:.4f}")
        print(f"{title} AUC: {roc_auc:.4f}")
'''


# Make fragment level predictions for the test set using the best model for each design
'''
design = sys.argv[1]
print(f"Running Model Evaluation for {design} model", flush=True)

# Load the trained model
api = wandb.Api()

if design == "CNN":
    run = api.run("christoffer-je-aarhus-university/cfDNA/3pvqujqd") # from sweep with configs (chosen rank)
    model = CNN_normalize_combined_end_motif(num_filters=run.config['num_filters'],
                                        kernel_size=run.config['kernel_size'],
                                        length_hidden_size=run.config['length_hidden_size'],
                                        fusion_hidden_size=run.config['fusion_hidden_size'])
    artifact = api.artifact("christoffer-je-aarhus-university/cfDNA/model:v68246") # Best config trained on train and val
    artifact_dir = artifact.download()
    model.load_state_dict(torch.load(f'{artifact_dir}/model.pth'))
    model.eval()

if design == "LSTM":
    run = api.run("christoffer-je-aarhus-university/cfDNA/4bv35gd2") # from sweep with configs (chosen rank)
    model = LSTM_BI_normalize_combined_end_motif(lstm_hidden_size=run.config['lstm_hidden_size'], 
                            length_hidden_size=run.config['length_hidden_size'],
                            fusion_hidden_size=run.config['fusion_hidden_size'])
    #artifact = api.artifact("christoffer-je-aarhus-university/cfDNA/model:v40206") # Best config trained on train and val
    artifact = api.artifact("christoffer-je-aarhus-university/cfDNA/model:v40128")
    artifact_dir = artifact.download()
    model.load_state_dict(torch.load(f'{artifact_dir}/model.pth'))
    model.eval()

if design == "GRU":
    run = api.run("christoffer-je-aarhus-university/cfDNA/lxt8g9y1") # from sweep with configs (chosen rank)
    model = GRU_BI_normalize_combined_end_motif(gru_hidden_size=run.config['gru_hidden_size'], 
                            length_hidden_size=run.config['length_hidden_size'],
                            fusion_hidden_size=run.config['fusion_hidden_size'])
    artifact = api.artifact("christoffer-je-aarhus-university/cfDNA/model:v40207") # Best config trained on train and val
    artifact_dir = artifact.download()
    model.load_state_dict(torch.load(f'{artifact_dir}/model.pth'))
    model.eval()

if design == "FNN":
    run = api.run("christoffer-je-aarhus-university/cfDNA/102p1irm") # from sweep with configs (chosen rank)
    model = FNN_normalize_combined_end_motif(end_motif_hidden_size=run.config['end_motif_hidden_size'],
                                  length_hidden_size=run.config['length_hidden_size'],
                                  fusion_hidden_size=run.config['fusion_hidden_size'])
    artifact = api.artifact("christoffer-je-aarhus-university/cfDNA/model:v68245") # Best config trained on train and val
    artifact_dir = artifact.download()
    model.load_state_dict(torch.load(f'{artifact_dir}/model.pth'))
    model.eval()
if design == "LogReg":
    run = wandb.init()
    artifact = run.use_artifact("christoffer-je-aarhus-university/cfDNA/model:v68248")
    artifact_dir = artifact.download()
    log_model = load(f'{artifact_dir}/logistic_regression_cv.joblib')
    model = WrappedLogRegModel(log_model)



# Getting test data
test_df = pd.read_csv("../data/test_from10.csv")
length_train_mean, length_train_std = np.load("normalization_stats/length_stats_kmer_combined_train_and_val_combined.npy")
test_dataset = cfDataset(test_df, mean=length_train_mean, std=length_train_std, normalize=True, kmer_combined=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

predictions = np.empty(len(test_dataloader))
with torch.no_grad():
    for i, (kmer, length, label, length_one_hot) in enumerate(test_dataloader):
        if i % 10000 == 0:
            print(f"Predicting sample {i}/{len(test_dataloader)}", flush=True)
        output = model(kmer, length, length_one_hot)
        if design == "LogReg":
            predictions[i] = output
        else:
            predictions[i] = torch.sigmoid(output).item()


np.save(
    f"../data/prediction_stats/best_{design}_test_fragment_level_predictions.npy",
    predictions
)
'''