import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
from utils import one_hot_encode_DNA_sequence, one_hot_encode_length, one_hot_encode_DNA_sequence_batch_numpy, one_hot_encode_length_batch
import os

class cfDataset(Dataset):
    '''Datset to store and access end-motifs, length and label from cell-free DNA reads'''
    
    def __init__(self, data: pd.DataFrame, mean=None, std=None, normalize:bool = True, kmer_combined:bool = False):
        '''Initilize the dataset by reading data from a csv file
        Args:
            data (pd.DataFrame): The dataframe containing the data for the dataset
            config (dict): The configuration for the dataset
        '''
        self.data = data
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.kmer_combined = kmer_combined

        # Saftey check: If normalize=True, mean and std must be provided
        if self.normalize and (self.mean is None or self.std is None):
            raise ValueError("Mean and std must be provided for normalization")        

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        sample = self.data.iloc[idx]
        label = torch.tensor(sample['class'], dtype=torch.float32)
        
        if self.kmer_combined:
            start_kmer = one_hot_encode_DNA_sequence(sample['start_kmer'])
            end_kmer = one_hot_encode_DNA_sequence(sample['end_kmer'])
            kmer = [start_kmer, end_kmer]
        else:
            kmer = one_hot_encode_DNA_sequence(sample['kmer'])

        if self.normalize:
            length = torch.tensor(sample['fragment_length'], dtype=torch.float32)
            length = (length - self.mean) / self.std
            length_one_hot = length  # Placeholder, not used when normalize is True
        else: 
            length_one_hot = one_hot_encode_length(length = sample['fragment_length'])
            length = length_one_hot  # Placeholder, not used when normalize is False
            

        return(kmer, length, label, length_one_hot)

        
class cfIterableDataset(IterableDataset):
    def __init__(self, file_path, mean=None, std=None, normalize=False, chunksize=10000):
        self.file_path = file_path
        self.normalize = normalize
        self.chunksize = chunksize
        self.mean = mean
        self.std = std

        # Infer label from path
        folder = os.path.basename(os.path.dirname(self.file_path))
        if folder == "pos":
            self.label_value = 1
        elif folder == "neg":
            self.label_value = 0
        else:
            raise ValueError("Could not infer label from file path")

        # Saftey check: If normalize=True, mean and std must be provided
        if self.normalize and (self.mean is None or self.std is None):
            raise ValueError("Mean and std must be provided for normalization")

    def __iter__(self):
        reader = pd.read_csv(
            self.file_path,
            sep="\t",
            compression="gzip",
            chunksize=self.chunksize
        )

        for chunk in reader:            
            # Remove rows with 'N' in either r1_kmer or r2_kmer
            mask = (
                ~chunk["r1_kmer"].str.contains("N") &
                ~chunk["r2_kmer"].str.contains("N") &
                (chunk["fragment_length"] >= 10) &
                (chunk["fragment_length"] <= 500)
            )
            chunk = chunk[mask]

            kmers_r1 = chunk["r1_kmer"].values
            kmers_r2 = chunk["r2_kmer"].values
            kmers_r1 = one_hot_encode_DNA_sequence_batch_numpy(kmers_r1)
            kmers_r2 = one_hot_encode_DNA_sequence_batch_numpy(kmers_r2)
            lengths = chunk["fragment_length"].values.copy()
            valid_lengths = (lengths >= 10) & (lengths <= 500)
            lengths = lengths[valid_lengths]
            lengths = torch.from_numpy(lengths).float()
            labels = torch.full((len(chunk),), self.label_value, dtype=torch.float32)

            if self.normalize:
                lengths = (lengths - self.mean) / self.std
                lengths_one_hot = lengths # Placeholder, not used when normalize is True

            else:
                lengths_one_hot = one_hot_encode_length_batch(lengths)
                lengths = lengths_one_hot # Placeholder, not used when normalize is False

            # Avoid yielding empty batches, which can happen if all rows in a chunk are invalid
            if len(kmers_r1) == 0:
                continue

            yield (
                kmers_r1,                                   # (B, k, 4)
                kmers_r2,                                   # (B, k, 4)
                lengths,                                    # (B,)
                labels,                                     # (B,)
                lengths_one_hot,                            # (B,) or (B, L)
            )
