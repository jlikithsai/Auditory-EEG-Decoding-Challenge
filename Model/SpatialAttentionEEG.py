# -*- coding: utf-8 -*-
"""SpatialAttention_eeg.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ra_QOOkNvTUikTwQlEkPlOjXgSJC2z_x
"""

#AttentionEncodermodule, which takes in an input sequence of EEG data and outputs a context vector that captures the relevant spatial and temporal features of the input sequence.


import torch
import torch.nn as nn

# Define the AttentionEncoder module
class AttentionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.attention = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, input_seq):
        # input_seq: (seq_len, batch_size, input_size)
        outputs, (hidden, cell) = self.lstm(input_seq)
        # outputs: (seq_len, batch_size, hidden_size * 2)
        # hidden: (2, batch_size, hidden_size)
        # cell: (2, batch_size, hidden_size)
        energy = self.attention(outputs)
        # energy: (seq_len, batch_size, 1)
        attention_weights = torch.softmax(energy, dim=0)
        # attention_weights: (seq_len, batch_size, 1)
        context_vector = torch.sum(outputs * attention_weights, dim=0)
        # context_vector: (batch_size, hidden_size * 2)
        output = torch.tanh(self.fc(context_vector))
        # output: (batch_size, hidden_size)
        return output


"""Example:
# Define the EEG data
eeg_data = torch.randn(10, 32, 64)  # (seq_len, batch_size, input_size)

# Initialize the AttentionEncoder module
input_size = 64
hidden_size = 128
ae = AttentionEncoder(input_size, hidden_size)

# Pass the EEG data through the AttentionEncoder module
output = ae(eeg_data)

# Print the output shape
print(output.shape)
print(output)"""