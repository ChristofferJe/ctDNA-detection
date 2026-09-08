import torch.nn as nn
import torch


class FNN_end_motif_only_individual_end_motif(nn.Module):
    '''Single-feature FNN implemented for end motif
        using the individual end motif setting. 
    '''

    def __init__(self, hidden_size: int, input_size: int = 40):
        super(FNN_end_motif_only_individual_end_motif, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        flattened_kmer = self.flatten(kmer)

        hidden_output = self.relu(self.hidden_layer(flattened_kmer))
 
        output = self.output_layer(hidden_output)

        return output
    
class FNN_end_motif_only_combined_end_motif(nn.Module):
    '''Single-feature FNN implemented for end motif
        using the combined end motif setting. 
    '''

    def __init__(self, hidden_size: int, input_size: int = 40):
        super(FNN_end_motif_only_combined_end_motif, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size*2, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        start_kmer, end_kmer = kmer
        flattened_start_kmer = self.flatten(start_kmer)
        flattened_end_kmer = self.flatten(end_kmer)

        start_kmer_flattened = self.relu(self.hidden_layer(flattened_start_kmer))
        end_kmer_flattened = self.relu(self.hidden_layer(flattened_end_kmer))
        
        combined_hidden_output = torch.cat((start_kmer_flattened, end_kmer_flattened), dim = 1)

        output = self.output_layer(combined_hidden_output)

        return output

    
class FNN_length_only_oh(nn.Module):
    '''Single-feature FNN implemented for fragment length
        using one-hot encoding for the length fragment preprocessing. 
    '''

    def __init__(self, hidden_size:int):
        super(FNN_length_only_oh, self).__init__()
        self.hidden_layer = nn.Linear(481, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        hidden_output = self.relu(self.hidden_layer(length_one_hot))
        output = self.output_layer(hidden_output)
        return output


class FNN_length_only_normalize(nn.Module):
    '''Single-feature FNN implemented for fragment length
        using normalization for the length fragment preprocessing. 
    '''

    def __init__(self, hidden_size:int):
        super(FNN_length_only_normalize, self).__init__()
        self.hidden_layer = nn.Linear(1, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        length = length.unsqueeze(1) 
        hidden_output = self.relu(self.hidden_layer(length))
        output = self.output_layer(hidden_output)
        return output



class FNN_normalize_combined_end_motif(nn.Module):
    '''Multimodal FNN implemented for the combined end motif setting
        and using normalization for the fragment length preprocessing.
    '''

    def __init__(self, end_motif_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int, input_size: int = 40):
        super(FNN_normalize_combined_end_motif, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.end_motif_hidden_layer = nn.Linear(input_size, end_motif_hidden_size)
        self.length_hidden_layer = nn.Linear(1, length_hidden_size)
        self.fusion_hidden_layer = nn.Linear(end_motif_hidden_size*2 + length_hidden_size, fusion_hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        start_kmer, end_kmer = kmer
        flattened_start_kmer = self.flatten(start_kmer)
        flattened_end_kmer = self.flatten(end_kmer)
        length = length.unsqueeze(1)

        start_kmer_output = self.relu(self.end_motif_hidden_layer(flattened_start_kmer))
        end_kmer_output = self.relu(self.end_motif_hidden_layer(flattened_end_kmer))
        
        length_output = self.relu(self.length_hidden_layer(length))

        features_combined = torch.cat((length_output, start_kmer_output, end_kmer_output), dim = 1)

        fusion_output = self.relu(self.fusion_hidden_layer(features_combined))

        output = self.output_layer(fusion_output)

        return output

class FNN_normalize_individual_end_motif(nn.Module):
    '''Multimodal FNN implemented for the individual end motif setting
        and using normalization for the fragment length preprocessing.
    '''

    def __init__(self, end_motif_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int, input_size: int = 40):
        super(FNN_normalize_individual_end_motif, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.end_motif_hidden_layer = nn.Linear(input_size, end_motif_hidden_size)
        self.length_hidden_layer = nn.Linear(1, length_hidden_size)
        self.fusion_hidden_layer = nn.Linear(end_motif_hidden_size + length_hidden_size, fusion_hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        flattened_kmer = self.flatten(kmer)
        length = length.unsqueeze(1)

        kmer_output = self.relu(self.end_motif_hidden_layer(flattened_kmer))

        length_output = self.relu(self.length_hidden_layer(length))

        features_combined = torch.cat((length_output, kmer_output), dim = 1)

        fusion_output = self.relu(self.fusion_hidden_layer(features_combined))

        output = self.output_layer(fusion_output)

        return output


class FNN_oh_combined_end_motif(nn.Module):
    '''Multimodal FNN implemented for the combined end motif setting
        and using one-hot encoding for the fragment length preprocessing.
    '''

    def __init__(self, end_motif_hidden_size:int, length_hidden_size: int, fusion_hidden_size: int, input_size: int = 40):
        super(FNN_oh_combined_end_motif, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.end_motif_hidden_layer = nn.Linear(input_size, end_motif_hidden_size)
        self.length_hidden_layer = nn.Linear(491, length_hidden_size)
        self.fusion_hidden_layer = nn.Linear(end_motif_hidden_size*2 + length_hidden_size, fusion_hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        start_kmer, end_kmer = kmer
        flattened_start_kmer = self.flatten(start_kmer)
        flattened_end_kmer = self.flatten(end_kmer)

        start_kmer_output = self.relu(self.end_motif_hidden_layer(flattened_start_kmer))
        end_kmer_output = self.relu(self.end_motif_hidden_layer(flattened_end_kmer))

        length_output = self.relu(self.length_hidden_layer(length_one_hot))

        features_combined = torch.cat((length_output, start_kmer_output, end_kmer_output), dim = 1)

        fusion_output = self.relu(self.fusion_hidden_layer(features_combined))

        output = self.output_layer(fusion_output)

        return output

class FNN_oh_individual_end_motif(nn.Module):
    '''Multimodal FNN implemented for the individual end motif setting
        and using one-hot encoding for the fragment length preprocessing.
    '''

    def __init__(self, end_motif_hidden_size:int, length_hidden_size: int, fusion_hidden_size: int, input_size: int = 40):
        super(FNN_oh_individual_end_motif, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.end_motif_hidden_layer = nn.Linear(input_size, end_motif_hidden_size)
        self.length_hidden_layer = nn.Linear(491, length_hidden_size)
        self.fusion_hidden_layer = nn.Linear(end_motif_hidden_size + length_hidden_size, fusion_hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        flattened_kmer = self.flatten(kmer)
        length = length.unsqueeze(1)

        kmer_output = self.relu(self.end_motif_hidden_layer(flattened_kmer))

        length_output = self.relu(self.length_hidden_layer(length_one_hot))

        features_combined = torch.cat((length_output, kmer_output), dim = 1)

        fusion_output = self.relu(self.fusion_hidden_layer(features_combined))

        output = self.output_layer(fusion_output)

        return output