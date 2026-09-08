from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from utils import build_feature_matrix, save, WrappedLogRegModel, generate_logreg_simulated_prediction_dataframe
from joblib import dump, load
import wandb

#alias pythoncf='LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python'

# Train and save logistic regression model
# Load training data
train_df = pd.read_csv("../data/train_and_val_from10.csv")

# Prepare features and labels
X_train = build_feature_matrix(train_df)
y_train = train_df["class"].values

# Define and fit logistic regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

run = wandb.init(entity = "christoffer-je-aarhus-university", 
                 project = "cfDNA",
                 settings=wandb.Settings(console="off"))
run.name = f"Fragment_level_logistic_regression"

# Save model to wandb
dump(log_model, "save/logistic_regression_cv.joblib")
artifcat = wandb.Artifact("model", type='model')
artifcat.add_file('save/logistic_regression_cv.joblib')
run.log_artifact(artifcat)


# Download traind logistic regression model from wandb
'''
run = wandb.init()
artifact = run.use_artifact("christoffer-je-aarhus-university/cfDNA/model:v68248")
artifact_dir = artifact.download()
log_model = load(f'{artifact_dir}/logistic_regression_cv.joblib')
'''

# Load test data and generate simulated predictions
test_df =  pd.read_csv("../data/test_from10.csv")

# Setting number of patients to simulate and different frac to use
positive_frac_list = [0.00, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02]
num_fragments_list = [100000]
num_patients = 100

wrapped_log_model = WrappedLogRegModel(log_model)

# Generate predictions
name = "logistic_regression_simulated_predictions_test"
for num_fragments in num_fragments_list:
    detection_df = generate_logreg_simulated_prediction_dataframe(wrapped_log_model, test_df, positive_frac_list, 
                                                           num_patients, num_fragments, 
                                                           name = name)


