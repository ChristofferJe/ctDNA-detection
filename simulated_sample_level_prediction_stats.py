from CNN.models import *
from RNN.models import *
from FNN.models import *
from utils import generate_simulated_prediction_dataframe
import pandas as pd
import torch
import wandb
import sys

# alias pythoncf='LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python'

# Load data
test_df =  pd.read_csv("../data/test_from10.csv")

design = sys.argv[1]
best_model = f"{design}"
name = f"best_{design}_individual_predictions"

# Load trained model
if best_model == "GRU":
    model = GRU_BI_normalize_combined_end_motif(gru_hidden_size=128, length_hidden_size=256, fusion_hidden_size=128)
    run = wandb.init()
    artifact = run.use_artifact("christoffer-je-aarhus-university/cfDNA/model:v40207")
if best_model == "LSTM":
    model = LSTM_BI_normalize_combined_end_motif(lstm_hidden_size=128, length_hidden_size=64, fusion_hidden_size=512)
    run = wandb.init()
    artifact = run.use_artifact("christoffer-je-aarhus-university/cfDNA/model:v40206")
if best_model == "CNN":
    model = CNN_normalize_combined_end_motif(num_filters=256, kernel_size=6, length_hidden_size=128, fusion_hidden_size=512)
    run = wandb.init()
    artifact = run.use_artifact("christoffer-je-aarhus-university/cfDNA/model:v68246")
if best_model == "FNN":
    model = FNN_normalize_combined_end_motif(end_motif_hidden_size=64, length_hidden_size=256, fusion_hidden_size=256)
    run = wandb.init()
    artifact = run.use_artifact("christoffer-je-aarhus-university/cfDNA/model:v68245")
artifact_dir = artifact.download()
model.load_state_dict(torch.load(f'{artifact_dir}/model.pth'))

# Setting number of patients to simulate and different frac to use
positive_frac_list = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02]
num_fragments_list = [100000]
num_patients = 100

# Generating prediction for the different number of fragments
for num_fragments in num_fragments_list:
    detection_df = generate_simulated_prediction_dataframe(model, test_df, positive_frac_list, 
                                                           num_patients, num_fragments, 
                                                           name = name)



    


    

