import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from FNN.models import *
from CNN.models import *
from RNN.models import *
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import copy
import os

# Initializing device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

def model_pipeline():
    '''Pipeline to train and evaluate a custom model with the given hyperparameters
    
    Args:
        run_name (str): Name of the wandb run
        hyperparameters (dict): Dictionary containing the hyperparameters for the model training and evaluation
        entity (str): The wandb entity (user or team) to which the run belongs
        project (str): The wandb project to which the run belongs
    '''
    # Initialize a new wandb run with the given name, entity, project and hyperparameters
    with wandb.init() as run:

        # Set the hyperparameters for the run to the config saved in wandb
        config = run.config

        # Change name of run to fit config 
        if config.architecture.startswith("FNN"):
            if config.data_type == "combined":
                run.name = f'run_{config.architecture}_end_motif_hidden-{config.end_motif_hidden_size}_length_hidden-{config.length_hidden_size}_lr-{config.learning_rate}_batch-{config.batch_size}_weight_decay-{config.weight_decay}'
            else:
                run.name = f'run_{config.architecture}_hidden-{config.hidden_size}_lr-{config.learning_rate}_batch-{config.batch_size}_weight_decay-{config.weight_decay}'
        if config.architecture.startswith("CNN"):
            run.name = f'run_{config.architecture}_num_filters-{config.num_filters}_kernel_size-{config.kernel_size}_length_hidden_size-{config.length_hidden_size}_lr-{config.learning_rate}_batch-{config.batch_size}_weight_decay-{config.weight_decay}'
        if config.architecture.startswith("GRU"):
            run.name = f'run_{config.architecture}_gru_hidden-{config.gru_hidden_size}_length_hidden_size-{config.length_hidden_size}_lr-{config.learning_rate}_batch-{config.batch_size}_weight_decay-{config.weight_decay}'
        if config.architecture.startswith("LSTM"):
            run.name = f'run_{config.architecture}_lstm_hidden-{config.lstm_hidden_size}_length_hidden_size-{config.length_hidden_size}_lr-{config.learning_rate}_batch-{config.batch_size}_weight_decay-{config.weight_decay}'

        # Initialize the model, dataloader, loss function and optimizer using the hyperparameters from the config
        model = build_model(config)
        model.to(device, non_blocking=True)
        train_dataloader, validation_dataloader = build_dataloaders(config)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, train_dataloader)
        loss_fn = build_loss(config)

        # Train the model using the train dataloader, loss function and optimizer
        train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, scheduler, config)

        # Save trained model to wandb
        save(model, run)

        # Evaluate the model on the test set
        evaluate(model, validation_dataloader, loss_fn)


def build_model(config: dict) -> torch.nn.Module:
    '''Builde the model as specified in the hyperparameters from the config
    
    Args:
        config (dict): Dictionary containing the hyperparameters for the model 

    Returns:
        torch.nn.Module: The initialized model as specified in the hyperparameters from the config
    '''
    if config.architecture == "FNN":
        if config.data_type == "combined":
            if config.normalize:
                if config.combined_end_motif:
                    model = FNN_normalize_combined_end_motif(end_motif_hidden_size=config.end_motif_hidden_size, 
                                                            length_hidden_size=config.length_hidden_size, 
                                                            fusion_hidden_size=config.fusion_hidden_size)
                else:
                    model = FNN_normalize_individual_end_motif(end_motif_hidden_size=config.end_motif_hidden_size, 
                                                            length_hidden_size=config.length_hidden_size, 
                                                            fusion_hidden_size=config.fusion_hidden_size)
            else:
                if config.combined_end_motif:
                    model = FNN_oh_combined_end_motif(end_motif_hidden_size=config.end_motif_hidden_size, 
                                                      length_hidden_size=config.length_hidden_size, 
                                                      fusion_hidden_size=config.fusion_hidden_size)
                else: 
                    model = FNN_oh_individual_end_motif(end_motif_hidden_size=config.end_motif_hidden_size, 
                                                        length_hidden_size=config.length_hidden_size, 
                                                        fusion_hidden_size=config.fusion_hidden_size)
        elif config.data_type == "length_only":
            if config.normalize:
                model = FNN_length_only_normalize(hidden_size=config.hidden_size)
            else:
                model = FNN_length_only_oh(hidden_size=config.hidden_size)
        elif config.data_type == "kmer_only":
            if config.combined_end_motif:
                model = FNN_end_motif_only_combined_end_motif(hidden_size=config.hidden_size)
            else:
                model = FNN_end_motif_only_individual_end_motif(hidden_size=config.hidden_size)


    if config.architecture == "CNN":
        if config.normalize:
            if config.combined_end_motif:
                model = CNN_normalize_combined_end_motif(num_filters=config.num_filters, kernel_size=config.kernel_size, 
                                    length_hidden_size=config.length_hidden_size, fusion_hidden_size=config.fusion_hidden_size)
            else:
                model = CNN_normalize_individual_end_motif(num_filters=config.num_filters, kernel_size=config.kernel_size, 
                                    length_hidden_size=config.length_hidden_size, fusion_hidden_size=config.fusion_hidden_size)
        else:
            if config.combined_end_motif:
                model = CNN_oh_combined_end_motif(num_filters=config.num_filters, kernel_size=config.kernel_size, 
                                    length_hidden_size=config.length_hidden_size, fusion_hidden_size=config.fusion_hidden_size)
            else:
                model = CNN_oh_individual_end_motif(num_filters=config.num_filters, kernel_size=config.kernel_size, 
                                    length_hidden_size=config.length_hidden_size, fusion_hidden_size=config.fusion_hidden_size)
    
    if config.architecture == "GRU_BI":
        if config.normalize:
            if config.combined_end_motif:
                model = GRU_BI_normalize_combined_end_motif(gru_hidden_size=config.gru_hidden_size, 
                                            length_hidden_size=config.length_hidden_size, 
                                            fusion_hidden_size=config.fusion_hidden_size)
            else:
                model = GRU_BI_normalize_individual_end_motif(gru_hidden_size=config.gru_hidden_size, 
                                length_hidden_size=config.length_hidden_size, 
                                fusion_hidden_size=config.fusion_hidden_size)
        else:
            if config.combined_end_motif:
                model = GRU_BI_oh_combined_end_motif(gru_hidden_size=config.gru_hidden_size, 
                                                    length_hidden_size=config.length_hidden_size, 
                                                    fusion_hidden_size=config.fusion_hidden_size)
            else:
                model = GRU_BI_oh_individual_end_motif(gru_hidden_size=config.gru_hidden_size, 
                                        length_hidden_size=config.length_hidden_size, 
                                        fusion_hidden_size=config.fusion_hidden_size)

        
    if config.architecture == "LSTM_BI":
        if config.normalize:
            if config.combined_end_motif:
                model = LSTM_BI_normalize_combined_end_motif(lstm_hidden_size=config.lstm_hidden_size, 
                                            length_hidden_size=config.length_hidden_size, 
                                            fusion_hidden_size=config.fusion_hidden_size)
            else:
                model = LSTM_BI_normalize_individual_end_motif(lstm_hidden_size=config.lstm_hidden_size, 
                                length_hidden_size=config.length_hidden_size, 
                                fusion_hidden_size=config.fusion_hidden_size)
        else:
            if config.combined_end_motif:
                model = LSTM_BI_oh_combined_end_motif(lstm_hidden_size=config.lstm_hidden_size, 
                                                    length_hidden_size=config.length_hidden_size, 
                                                    fusion_hidden_size=config.fusion_hidden_size)
            else:
                model = LSTM_BI_oh_individual_end_motif(lstm_hidden_size=config.lstm_hidden_size, 
                                        length_hidden_size=config.length_hidden_size, 
                                        fusion_hidden_size=config.fusion_hidden_size)
    return model


def build_dataloaders(config: dict) -> torch.utils.data.DataLoader:
    '''Build a dataloader for the model training as specified by the hyperparameters from the config
    
    Args:
        config (dict): Dictionary containing the hyperparameters for the model 

    Returns: 
        torch.utils.data.DataLoader: The initialized dataloaders for the model training and testing
    '''
    from cfDataset import cfDataset # to aviod circular imports, as cfDataset also imports one_hot_encode_DNA_sequence from utils.py
    
    # Load the training and validation data
    train_length_mean, train_length_std = None, None 
    if config.combined_end_motif:
        train_data = pd.read_csv("../data/train_from10.csv")
        validation_data = pd.read_csv("../data/val_from10.csv")
        if config.normalize:
            # Load mean and std of length from training data for normalization
            train_length_mean, train_length_std = np.load("normalization_stats/length_stats_kmer_combined.npy")
    else:
        train_data = pd.read_csv("../data/train_longer_from10.csv")
        validation_data = pd.read_csv("../data/val_longer_from10.csv")
        if config.normalize:
            # Load mean and std of length from training data for normalization
            train_length_mean, train_length_std = np.load("normalization_stats/length_stats.npy")
        

    # Initialize the train and validation datasets
    train_dataset = cfDataset(train_data, mean=train_length_mean, std=train_length_std, normalize=config.normalize, kmer_combined=config.combined_end_motif)
    validation_dataset = cfDataset(validation_data, mean=train_length_mean, std=train_length_std, normalize=config.normalize, kmer_combined=config.combined_end_motif)

    # Initialize the train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    return train_dataloader, validation_dataloader

def build_optimizer(config: dict, model: torch.nn.Module):
    '''Build an optimizer for the model training as specified by the hyperparameters from the config
    
    Args:
        config (dict): Dictionary containing the hyperparameters for the model
        model (torch.nn.Module): The model for which the optimizer is to be built

    Returns:
        torch.optim.Optimizer: The initialized optimizer for the model training
    '''
    if config.optimizer == "Adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)

    return optimizer

def build_scheduler(config: dict, optimizer, train_dataloader):
    '''Builde learning rate scheduler as specified in the config file
    
    Args: 
        config (dict): The configuration file specifing the specific run
        optimizer: The optimizer for which the scheduler is being build
        train_dataloader: The dataloader which is going to be used for the training together with the optimizer
        
    Returns:
        The build scheduler
    '''
    steps_per_epoch = len(train_dataloader)
    total_steps = config.epochs * steps_per_epoch

    if config.scheduler == "None":
        scheduler = "None"
    if config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=total_steps, 
                                                               eta_min=config.scheduler_lr_min)
    if config.scheduler == "CyclicalLR":
        step_size_up = 2*steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                      base_lr=config.scheduler_lr_min,
                                                      max_lr=config.scheduler_lr_max,
                                                      step_size_up=step_size_up,
                                                      cycle_momentum=optimizer_with_momentum(config))
    if config.scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=config.scheduler_lr_max,
                                                        total_steps=total_steps,
                                                        cycle_momentum=optimizer_with_momentum(config),
                                                        pct_start=0.3,
                                                        anneal_strategy="cos")
    
    return scheduler
                                                    

def optimizer_with_momentum(config: dict):
    '''Checks if the chosen optimizer is using momentum'''
    if config.optimizer == "RMSprop" or config.optimizer == "SGD":
        return True
    else:
        return False

def build_loss(config: dict):
    '''Build a loss function for the model training as specified by the hyperparameters from the config
    
    Args:
        config (dict): Dictionary containing the hyperparameters for the model

    Returns:
        torch.nn.Module: The initialized loss function for the model training
    '''
    if config.loss == "BCE":
        loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    return loss


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, validation_dataloader: torch.utils.data.DataLoader, 
          loss_fn , optimizer, scheduler, config: dict):
    '''Function to train the model using the given dataloader, loss function and optimizer for a specified number of epochs
    
    Args:
        model (torch.nn.Module): The model to be trained
        train_data_loader (torch.utils.data.DataLoader): The dataloader for the training data
        test_data_loader(torch.utils.data.DataLoader): The dataloader for the validation data
        loss_fn: The loss function for the model training
        optimizer: The optimizer for the model training
        scheduler: The scheduler used to modify the learning rate of the optimizer
        config (dict): Dictionary containing the hyperparameters for the model training
    '''
    model.train()
    batch_count = 0


    # Validate model before training
    val_loss = validate(model = model, validation_dataloader=validation_dataloader, loss_fn=loss_fn) 

    # Initialize early stopping if enabled in config
    if True: #config.early_stopping:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = 0

    for epoch in tqdm(range(config.epochs)):
        model.train()
        total_loss = 0
        for kmer, length, label, length_one_hot in train_dataloader:
            if isinstance(kmer, torch.Tensor):
                kmer = kmer.to(device, non_blocking=True)
            elif isinstance(kmer, list):
                kmer = [t.to(device, non_blocking=True) for t in kmer]
            length = length.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            length_one_hot = length_one_hot.to(device, non_blocking=True)

            batch_count += 1

            output = model(kmer = kmer, length = length, length_one_hot = length_one_hot)
            
            label = label.float()
            loss = loss_fn(output, label.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config.scheduler != "None":
                scheduler.step()

            total_loss += loss.item()

            # Log the loss to wandb 
            if batch_count % config.log_interval == 0:
                wandb.log({"loss": loss.item()})
        wandb.log({"epoch_loss": total_loss / len(train_dataloader)})

        # Validate model after each epoch of training 
        val_loss = validate(model = model, validation_dataloader=validation_dataloader, loss_fn=loss_fn) 

        if config.early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1 # +1 because epoch starts at 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}")
                wandb.log({"best_epoch": best_epoch})
                wandb.log({"best_val_loss": best_val_loss})
                wandb.log({"early_stopping_epoch": epoch + 1})
                model.load_state_dict(best_model_state)
                break


def save(model: torch.nn.Module, run):
    '''Save model to wandb as an artifact
    
    Args: 
        model (torch.nn.Module): The model to save
        run: The wandb run which the model is saved to
    '''
    torch.save(model.state_dict(), "save/model.pth")
    artifcat = wandb.Artifact("model", type='model')
    artifcat.add_file('save/model.pth')
    run.log_artifact(artifcat)

def validate(model: torch.nn.Module, validation_dataloader: torch.utils.data.DataLoader, loss_fn):
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for kmer, length, label, length_one_hot in validation_dataloader:
            if isinstance(kmer, torch.Tensor):
                kmer = kmer.to(device, non_blocking=True)
            elif isinstance(kmer, list):
                kmer = [t.to(device, non_blocking=True) for t in kmer]
            length = length.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            length_one_hot = length_one_hot.to(device, non_blocking=True)

            label = label.unsqueeze(1)

            # Get the model output for the batch
            output = model(kmer = kmer, length = length, length_one_hot = length_one_hot)

            # Calculate the loss for the batch and add to validation loss
            loss = loss_fn(output, label)
            validation_loss += loss.item()

        # Log validation to wandb
        avg_val_loss = validation_loss / len(validation_dataloader)
        wandb.log({"val_loss": avg_val_loss})

        return avg_val_loss
    
            
def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn):
    '''Function to evaluate the model on the test set and log the results to wandb
    
    Args: 
        model (torch.nn.Module): The model to be evaluated
        data_loader (torch.utils.data.DataLoader): The dataloader for the test set
        loss_fn: The loss function for the model evaluation
        config (dict): Dictionary containing the hyperparameters for the model training
    '''
    model.eval()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    all_labels = []
    all_probs = []
    with torch.no_grad():
        for kmer, length, label, length_one_hot in dataloader:
            if isinstance(kmer, torch.Tensor):
                kmer = kmer.to(device, non_blocking=True)
            elif isinstance(kmer, list):
                kmer = [t.to(device, non_blocking=True) for t in kmer]
            length = length.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            length_one_hot = length_one_hot.to(device, non_blocking=True)

            label = label.unsqueeze(1)

            # Get the model output for the batch
            output = model(kmer = kmer, length = length, length_one_hot = length_one_hot)

            # Calculate TP(True Positives), FP(False Positives), TN(True Negatives) and FN(False Negatives) for the batch
            prob = torch.sigmoid(output)
            output_binary = (prob > 0.5)
            TP += ((output_binary == 1) & (label == 1)).sum().item() 
            FP += ((output_binary == 1) & (label == 0)).sum().item() 
            TN += ((output_binary == 0) & (label == 0)).sum().item() 
            FN += ((output_binary == 0) & (label == 1)).sum().item() 

            # Save the labels and predicted probabilities for the batch
            all_labels.append(label.cpu())
            all_probs.append(prob.cpu())

    # Calculate accuracy, sensitivity and specificity
    accuracy = (TP + TN) / (TP + TN + FP + FN) 
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Calculate the AUC-ROC score
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    auc = roc_auc_score(all_labels, all_probs)

    # Log the mertrics to wandb
    wandb.log({"AUC": auc, "TP": TP, "TN": TN, "FP": FP, "FN": FN, 
               "acc": accuracy, "sens": sensitivity, "spec": specificity})


def re_train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, validation_dataloader: torch.utils.data.DataLoader, 
          loss_fn , optimizer, scheduler, epochs): 
    '''Function to train the model using the given dataloader, loss function and optimizer for a specified number of epochs
    
    Args:
        model (torch.nn.Module): The model to be trained
        train_data_loader (torch.utils.data.DataLoader): The dataloader for the training data
        test_data_loader(torch.utils.data.DataLoader): The dataloader for the validation data
        loss_fn: The loss function for the model training
        optimizer: The optimizer for the model training
        scheduler: The scheduler used to modify the learning rate of the optimizer
        epochs: the number of epochs to train the model for 
    '''
    model.train()
    batch_count = 0


    # Validate model before training
    val_loss = validate(model = model, validation_dataloader=validation_dataloader, loss_fn=loss_fn) 

    # Initialize early stopping if enabled in config
    if True: 
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for kmer, length, label, length_one_hot in train_dataloader:
            if isinstance(kmer, torch.Tensor):
                kmer = kmer.to(device, non_blocking=True)
            elif isinstance(kmer, list):
                kmer = [t.to(device, non_blocking=True) for t in kmer]
            length = length.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            length_one_hot = length_one_hot.to(device, non_blocking=True)

            batch_count += 1

            output = model(kmer = kmer, length = length, length_one_hot = length_one_hot)
            
            label = label.float()
            loss = loss_fn(output, label.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler != "None":
                scheduler.step()

            total_loss += loss.item()

            # Log the loss to wandb 
            if batch_count % 500 == 0: 
                wandb.log({"loss": loss.item()})
        wandb.log({"epoch_loss": total_loss / len(train_dataloader)})

        # Validate model after each epoch of training 
        val_loss = validate(model = model, validation_dataloader=validation_dataloader, loss_fn=loss_fn) 

        #Eearly stopping based on validation loss
        if True: 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1 # +1 because epoch starts at 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 10: 
                print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}")
                wandb.log({"best_epoch": best_epoch})
                wandb.log({"best_val_loss": best_val_loss})
                wandb.log({"early_stopping_epoch": epoch + 1})
                model.load_state_dict(best_model_state)
                break

def one_hot_encode_DNA_sequence(sequence: str) -> torch.Tensor:
    '''One-hot encode a DNA sequence
    
    Args:
        sequence (str): DNA sequence to be one-hot encoded

    Returns:
        torch.Tensor: One-hot encoded tensor of the DNA sequence with shape (sequence length, 4)
    '''

    # Define a mapping from nucleotides to integer indices
    mapping = {"A":0, "C":1, "G":2, "T":3}

    # Convert the sequence to a tensor of integer indices using the mapping
    indices = torch.tensor([mapping[nucleotide] for nucleotide in sequence])

    # One-hot encode the indices using PyTorch's built-in function
    one_hot_encoded = F.one_hot(indices, num_classes=4).float()

    return one_hot_encoded

def one_hot_encode_length(length: int, min:int = 10, max:int = 500) -> torch.Tensor:
    num_classes = max - min + 1

    shifted_length = torch.tensor(length - min, dtype=torch.long)

    return F.one_hot(shifted_length, num_classes=num_classes).float()

def one_hot_encode_length_batch(lengths, min_len=10, max_len=500):
    # Number of classes
    num_classes = max_len - min_len + 1

    # Convert to tensor if needed
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.from_numpy(lengths)

    # Shift to start at 0
    indices = lengths.long() - min_len

    return F.one_hot(indices, num_classes=num_classes).float()


def one_hot_encode_DNA_sequence_batch_numpy(kmers: list) -> torch.Tensor:
    lookup = np.zeros((128, 4), dtype=np.float32)
    lookup[ord('A')] = [1,0,0,0]
    lookup[ord('C')] = [0,1,0,0]
    lookup[ord('G')] = [0,0,1,0]
    lookup[ord('T')] = [0,0,0,1]

    arr = np.frombuffer("".join(kmers).encode(), dtype=np.uint8)
    arr = arr.reshape(len(kmers), -1)
    return torch.tensor(lookup[arr])

def generate_patient_dataloader(data: pd.DataFrame, num_fragments: int=5000, positive_rate: float=0.0):
    '''Generates a dataset which simulates a patient with a given rate of positive fragments
    
    Args:
        data (pd.DataFrame): The dataframe containing all fragments
        num_fragments (int): The total number of fragments to include in the dataset
        positive_rate (float): The rate of positive samples in the dataset
        
    Returns
        torch.Dataset: A dataset containin "num_fragments" fragments with a "positive_rate" of positive samples
    '''
    from cfDataset import cfDataset # to aviod circular imports, as cfDataset also imports one_hot_encode_DNA_sequence from utils.py

    # Get number of positive and negative samples
    num_negative = int(round(num_fragments * (1 - positive_rate)))
    num_positive = int(round(num_fragments * positive_rate))

    # Sampling the rows that will make up the simulated patient
    negative_samples = data[data['class'] == 0].sample(num_negative)
    positive_samples = data[data['class'] == 1].sample(num_positive)

    # Combining the positive and negative samples into one data frame
    patient_df = pd.concat([negative_samples, positive_samples])

    # Load mean and std for normalization
    mean_length, std_length = np.load("normalization_stats/length_stats_kmer_combined_train_and_val_combined.npy")

    # Use the simulated patient data frame to create a cfDataset 
    patient_dataset = cfDataset(patient_df, mean = mean_length, std = std_length, 
                                normalize=True, kmer_combined=True)

    # Generate a dataloader based on the patient dataset
    patient_dataloader = DataLoader(patient_dataset, shuffle=True, batch_size=1)

    return patient_dataloader


def generate_simulated_prediction_dataframe(model, test_df, positive_frac,
                                 num_patients, num_fragments=5000, name="None"):
    # Define the total number of observations in the data frame
    total_runs = len(positive_frac) * num_patients

    # Preallocate arrays
    pos_frac = np.empty(total_runs)
    frac_over_50 = np.empty(total_runs)
    frac_over_60 = np.empty(total_runs)
    frac_over_70 = np.empty(total_runs)
    frac_over_80 = np.empty(total_runs)
    frac_over_90 = np.empty(total_runs)
    frac_over_95 = np.empty(total_runs)
    mean_prob = np.empty(total_runs)

    idx = 0

    for frac in positive_frac:
        print(f"Starting positive frac: {frac}")
        
        for _ in tqdm(range(num_patients)):
            pos_frac[idx] = frac

            patient_dataloader = generate_patient_dataloader(
                data=test_df,
                num_fragments=num_fragments,
                positive_rate=frac
            )

            # Preallocate predictions
            predictions = np.empty(len(patient_dataloader))

            with torch.no_grad():
                for i, (kmer, length, label, length_one_hot) in enumerate(patient_dataloader):
                    output = model(kmer, length, length_one_hot)
                    predictions[i] = torch.sigmoid(output).item()

            # Vectorized computations
            frac_over_50[idx] = np.mean(predictions > 0.5)
            frac_over_60[idx] = np.mean(predictions > 0.6)
            frac_over_70[idx] = np.mean(predictions > 0.7)
            frac_over_80[idx] = np.mean(predictions > 0.8)
            frac_over_90[idx] = np.mean(predictions > 0.9)
            frac_over_95[idx] = np.mean(predictions > 0.95)
            mean_prob[idx] = np.mean(predictions)

            idx += 1

    # Build DataFrame (no conversion needed)
    df_results = pd.DataFrame({
        "pos_frac": pos_frac,
        "class": (pos_frac > 0).astype(int),
        "frac_over_50": frac_over_50,
        "frac_over_60": frac_over_60,
        "frac_over_70": frac_over_70,
        "frac_over_80": frac_over_80,
        "frac_over_90": frac_over_90,
        "frac_over_95": frac_over_95,
        "mean_prob": mean_prob
    })

    # Save to CSV
    output_csv = f'../data/prediction_stats/{name}.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return df_results


def generate_logreg_simulated_prediction_dataframe(model, test_df, positive_frac,
                                 num_patients, num_fragments=5000, name="None"):
    # Define the total number of observations in the data frame
    total_runs = len(positive_frac) * num_patients

    # Preallocate arrays
    pos_frac = np.empty(total_runs)
    frac_over_50 = np.empty(total_runs)
    frac_over_60 = np.empty(total_runs)
    frac_over_70 = np.empty(total_runs)
    frac_over_80 = np.empty(total_runs)
    frac_over_90 = np.empty(total_runs)
    frac_over_95 = np.empty(total_runs)
    mean_prob = np.empty(total_runs)

    idx = 0

    for frac in positive_frac:
        print(f"Starting positive frac: {frac}")
        
        for _ in tqdm(range(num_patients)):
            pos_frac[idx] = frac

            patient_dataloader = generate_patient_dataloader(
                data=test_df,
                num_fragments=num_fragments,
                positive_rate=frac
            )

            # Preallocate predictions
            predictions = np.empty(len(patient_dataloader))

            for i, (kmer, length, label, length_one_hot) in enumerate(patient_dataloader):
                output = model(kmer, length, length_one_hot)
                predictions[i] = output

            # Vectorized computations
            frac_over_50[idx] = np.mean(predictions > 0.5)
            frac_over_60[idx] = np.mean(predictions > 0.6)
            frac_over_70[idx] = np.mean(predictions > 0.7)
            frac_over_80[idx] = np.mean(predictions > 0.8)
            frac_over_90[idx] = np.mean(predictions > 0.9)
            frac_over_95[idx] = np.mean(predictions > 0.95)
            mean_prob[idx] = np.mean(predictions)

            idx += 1

    # Build DataFrame (no conversion needed)
    df_results = pd.DataFrame({
        "pos_frac": pos_frac,
        "class": (pos_frac > 0).astype(int),
        "frac_over_50": frac_over_50,
        "frac_over_60": frac_over_60,
        "frac_over_70": frac_over_70,
        "frac_over_80": frac_over_80,
        "frac_over_90": frac_over_90,
        "frac_over_95": frac_over_95,
        "mean_prob": mean_prob
    })

    # Save to CSV
    output_csv = f'../data/prediction_stats/{name}.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return df_results


def build_feature_matrix(df):
    X_list = []
    # load mean and std for length from training data for normalization
    mean_length, std_length = np.load("normalization_stats/length_stats_kmer_combined_train_and_val_combined.npy")
    for _, row in tqdm(df.iterrows()):
        start_kmer = row['start_kmer']
        end_kmer = row['end_kmer']
        length = row['fragment_length']


        # One hot endcode kmer
        start_kmer_one_hot = one_hot_encode_DNA_sequence(start_kmer)
        end_kmer_one_hot = one_hot_encode_DNA_sequence(end_kmer)

        # Flatten one-hot kmer
        start_kmer_one_hot_flat = start_kmer_one_hot.view(-1)
        end_kmer_one_hot_flat = end_kmer_one_hot.view(-1)
        

        # Convert to numoy
        start_kmer_one_hot_numpy = start_kmer_one_hot_flat.numpy()
        end_kmer_one_hot_numpy = end_kmer_one_hot_flat.numpy()

        # Normalize length
        length_normalized = (length - mean_length) / std_length 

        # Append length 
        features = np.concatenate([start_kmer_one_hot_numpy, end_kmer_one_hot_numpy, [length_normalized]])

        X_list.append(features)

    X = np.vstack(X_list)
    return X


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, start_kmer, end_kmer, length):

        kmer = [start_kmer, end_kmer]

        dummy_length_one_hot = length

        return self.model(kmer, length, dummy_length_one_hot)
    
class WrappedLogRegModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, kmer, length, length_one_hot):

        start_kmer, end_kmer = kmer

        dummy_length_one_hot = length

        # Now concatenate the features in the same way as in build_feature_matrix
        start_kmer_one_hot_flat = start_kmer.view(-1)
        end_kmer_one_hot_flat = end_kmer.view(-1)

        length_value = np.array([length.item()])

        start_kmer_one_hot_numpy = start_kmer_one_hot_flat.numpy()
        end_kmer_one_hot_numpy = end_kmer_one_hot_flat.numpy()

        # Append length 
        features = np.concatenate([start_kmer_one_hot_numpy, end_kmer_one_hot_numpy, length_value])
        features = features.reshape(1, -1)

        return self.model.predict_proba(features)[0,1]

def dataset_to_tensors(dataset):

    start_kmers = []
    end_kmers = []
    lengths = []
    labels = []

    for i in range(len(dataset)):

        kmer, length, label, _ = dataset[i]

        start_kmers.append(kmer[0])
        end_kmers.append(kmer[1])
        lengths.append(length)
        labels.append(label)

    start_kmers = torch.stack(start_kmers)
    end_kmers = torch.stack(end_kmers)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)

    return start_kmers, end_kmers, lengths, labels