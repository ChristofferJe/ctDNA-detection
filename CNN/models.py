import torch.nn as nn
import torch

class CNN_normalize_individual_end_motif(nn.Module):
    ''' Multimodal CNN for the individual end motif setting 
        and using normalization for the fragment length preprocessing.
    '''

    def __init__(self, num_filters: int, kernel_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(CNN_normalize_individual_end_motif, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.length_hidden_layer = nn.Linear(1, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear(num_filters * 10 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        kmer = kmer.transpose(1, 2).contiguous()
        conv_output = self.conv_layer(kmer)
        conv_output = conv_output.permute(0,2,1)
        conv_output = self.relu(conv_output)
        conv_output_flat = conv_output.flatten(start_dim=1)
        
        length = length.unsqueeze(1)
        length_output = self.relu(self.length_hidden_layer(length))

        fusion_input = torch.cat((conv_output_flat, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output


class CNN_oh_individual_end_motif(nn.Module):
    ''' Multimodal CNN for the individual end motif setting 
        and using one-hot encoding for the fragment length preprocessing.
    '''

    def __init__(self, num_filters: int, kernel_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(CNN_oh_individual_end_motif, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.length_hidden_layer = nn.Linear(491, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear(num_filters * 10 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        kmer = kmer.transpose(1, 2).contiguous()
        conv_output = self.conv_layer(kmer)
        conv_output = conv_output.permute(0,2,1)
        conv_output = self.relu(conv_output)
        conv_output_flat = conv_output.flatten(start_dim=1)

        length_output = self.relu(self.length_hidden_layer(length_one_hot))

        fusion_input = torch.cat((conv_output_flat, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output

class CNN_normalize_combined_end_motif(nn.Module):
    ''' Multimodal CNN for the combined end motif setting 
        and using normalization for the fragment length preprocessing.
    '''

    def __init__(self, num_filters: int, kernel_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(CNN_normalize_combined_end_motif, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.length_hidden_layer = nn.Linear(1, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear((num_filters * 10) * 2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: list, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        start_kmer, end_kmer = kmer
        start_kmer = start_kmer.transpose(1, 2).contiguous()
        end_kmer = end_kmer.transpose(1, 2).contiguous()
        conv_output1 = self.conv_layer(start_kmer)
        conv_output2 = self.conv_layer(end_kmer)
        conv_output1 = conv_output1.permute(0,2,1)
        conv_output2 = conv_output2.permute(0,2,1)
        conv_output1 = self.relu(conv_output1)
        conv_output2 = self.relu(conv_output2)
        conv_output_flat1 = conv_output1.flatten(start_dim=1)
        conv_output_flat2 = conv_output2.flatten(start_dim=1)
        
        length = length.unsqueeze(1)
        length_output = self.relu(self.length_hidden_layer(length))

        fusion_input = torch.cat((conv_output_flat1, conv_output_flat2, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output
    

class CNN_oh_combined_end_motif(nn.Module):
    ''' Multimodal CNN for the combined end motif setting 
        and using one-hot encoding for the fragment length preprocessing.
    '''

    def __init__(self, num_filters: int, kernel_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(CNN_oh_combined_end_motif, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.length_hidden_layer = nn.Linear(491, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear((num_filters * 10) * 2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: list, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        start_kmer, end_kmer = kmer
        start_kmer = start_kmer.transpose(1, 2).contiguous()
        end_kmer = end_kmer.transpose(1, 2).contiguous()
        conv_output1 = self.conv_layer(start_kmer)
        conv_output2 = self.conv_layer(end_kmer)
        conv_output1 = conv_output1.permute(0,2,1)
        conv_output2 = conv_output2.permute(0,2,1)
        conv_output1 = self.relu(conv_output1)
        conv_output2 = self.relu(conv_output2)
        conv_output_flat1 = conv_output1.flatten(start_dim=1)
        conv_output_flat2 = conv_output2.flatten(start_dim=1)
        
        length_output = self.relu(self.length_hidden_layer(length_one_hot))

        fusion_input = torch.cat((conv_output_flat1, conv_output_flat2, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output