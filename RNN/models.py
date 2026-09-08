import torch.nn as nn
import torch


class GRU_BI_normalize_individual_end_motif(nn.Module):
    ''' Multimodal BI GRU for the individual end motif setting 
        and using normalization for the fragment length preprocessing.
    '''

    def __init__(self, gru_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(GRU_BI_normalize_individual_end_motif, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=gru_hidden_size, batch_first=True, bidirectional=True)
        self.length_hidden_layer = nn.Linear(1, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear(gru_hidden_size * 2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(kmer) 
        h = h.transpose(0,1).reshape(h.size(1), -1) 
 
        length = length.unsqueeze(1)
        length_output = self.relu(self.length_hidden_layer(length))

        fusion_input = torch.cat((h, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output


class GRU_BI_oh_individual_end_motif(nn.Module):
    ''' Multimodal BI GRU for the individual end motif setting 
        and using one-hot encoding for the fragment length preprocessing.
    '''

    def __init__(self, gru_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(GRU_BI_oh_individual_end_motif, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=gru_hidden_size, batch_first=True, bidirectional=True)
        self.length_hidden_layer = nn.Linear(491, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear(gru_hidden_size * 2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(kmer) 
        h = h.transpose(0,1).reshape(h.size(1), -1) 
 
        length_output = self.relu(self.length_hidden_layer(length_one_hot))

        fusion_input = torch.cat((h, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output


class GRU_BI_normalize_combined_end_motif(nn.Module):
    ''' Multimodal BI GRU for the combined end motif setting 
        and using normalization for the fragment length preprocessing.
    '''

    def __init__(self, gru_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(GRU_BI_normalize_combined_end_motif, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=gru_hidden_size, batch_first=True, bidirectional=True)
        self.length_hidden_layer = nn.Linear(1, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear((gru_hidden_size * 2) * 2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        start_kmer, end_kmer = kmer
        _, h_start = self.gru(start_kmer) 
        _, h_end = self.gru(end_kmer) 
        h = torch.cat((h_start, h_end), dim=2)
        h = h.transpose(0,1).reshape(h.size(1), -1) 
 
        length = length.unsqueeze(1)
        length_output = self.relu(self.length_hidden_layer(length))

        fusion_input = torch.cat((h, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output


class GRU_BI_oh_combined_end_motif(nn.Module):
    ''' Multimodal BI GRU for the combined end motif setting 
        and using one-hot encoding for the fragment length preprocessing.
    '''

    def __init__(self, gru_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(GRU_BI_oh_combined_end_motif, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=gru_hidden_size, batch_first=True, bidirectional=True)
        self.length_hidden_layer = nn.Linear(491, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear((gru_hidden_size * 2) * 2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        start_kmer, end_kmer = kmer
        _, h_start = self.gru(start_kmer) 
        _, h_end = self.gru(end_kmer) 
        h = torch.cat((h_start, h_end), dim=2) 
        h = h.transpose(0,1).reshape(h.size(1), -1) 
 
        length_output = self.relu(self.length_hidden_layer(length_one_hot))

        fusion_input = torch.cat((h, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output

class LSTM_BI_normalize_individual_end_motif(nn.Module):
    ''' Multimodal BI LSTM for the individual end motif setting 
        and using normalization for the fragment length preprocessing.
    '''

    def __init__(self, lstm_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(LSTM_BI_normalize_individual_end_motif, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)
        self.length_hidden_layer = nn.Linear(1, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear(lstm_hidden_size*2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.lstm(kmer) 
        h = h.transpose(0,1).reshape(h.size(1), -1) 

        length = length.unsqueeze(1)
        length_output = self.relu(self.length_hidden_layer(length))

        fusion_input = torch.cat((h, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output

class LSTM_BI_oh_individual_end_motif(nn.Module):
    ''' Multimodal BI LSTM for the individual end motif setting 
        and using one-hot encoding for the fragment length preprocessing.
    '''

    def __init__(self, lstm_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(LSTM_BI_oh_individual_end_motif, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)
        self.length_hidden_layer = nn.Linear(491, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear(lstm_hidden_size*2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.lstm(kmer) 
        h = h.transpose(0,1).reshape(h.size(1), -1) 

        length_output = self.relu(self.length_hidden_layer(length_one_hot))

        fusion_input = torch.cat((h, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output

class LSTM_BI_normalize_combined_end_motif(nn.Module):
    ''' Multimodal BI LSTM for the combined end motif setting 
        and using normalization for the fragment length preprocessing.
    '''

    def __init__(self, lstm_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(LSTM_BI_normalize_combined_end_motif, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)
        self.length_hidden_layer = nn.Linear(1, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear((lstm_hidden_size*2)*2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        start_kmer, end_kmer = kmer
        _, (h_start, c_start) = self.lstm(start_kmer) 
        _, (h_end, c_end) = self.lstm(end_kmer) 
        h = torch.cat((h_start, h_end), dim=2)
        h = h.transpose(0,1).reshape(h.size(1), -1) 

        length = length.unsqueeze(1)
        length_output = self.relu(self.length_hidden_layer(length))

        fusion_input = torch.cat((h, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output


class LSTM_BI_oh_combined_end_motif(nn.Module):
    ''' Multimodal BI LSTM for the combined end motif setting 
        and using one-hot encoding for the fragment length preprocessing.
    '''

    def __init__(self, lstm_hidden_size: int, length_hidden_size: int, fusion_hidden_size: int):
        super(LSTM_BI_oh_combined_end_motif, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)
        self.length_hidden_layer = nn.Linear(491, length_hidden_size)
        self.relu = nn.ReLU()
        self.fusion_hidden_layer = nn.Linear((lstm_hidden_size*2)*2 + length_hidden_size, fusion_hidden_size)
        self.output_layer = nn.Linear(fusion_hidden_size, 1)

    def forward(self, kmer: torch.Tensor, length: torch.Tensor, length_one_hot: torch.Tensor) -> torch.Tensor:
        start_kmer, end_kmer = kmer
        _, (h_start, c_start) = self.lstm(start_kmer) 
        _, (h_end, c_end) = self.lstm(end_kmer) 
        h = torch.cat((h_start, h_end), dim=2)
        h = h.transpose(0,1).reshape(h.size(1), -1) 

        length_output = self.relu(self.length_hidden_layer(length_one_hot))

        fusion_input = torch.cat((h, length_output), dim=1)

        fusion_output = self.relu(self.fusion_hidden_layer(fusion_input))
        output = self.output_layer(fusion_output)

        return output