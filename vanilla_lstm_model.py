import torch
import torch.nn as nn

class VanillaLSTM(nn.Module):
    """
    A simple baseline LSTM model for trajectory prediction.
    It uses an encoder-decoder architecture.
    """
    def __init__(self, embedding_dim=64, hidden_dim=64, num_layers=1):
        """
        Args:
        - embedding_dim: The dimension of the input embedding.
        - hidden_dim: The dimension of the LSTM's hidden state.
        - num_layers: The number of layers in the LSTM.
        """
        super(VanillaLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input embedding layer
        # Maps the 2D coordinates to a higher-dimensional space
        self.embedding = nn.Linear(2, embedding_dim)
        
        # LSTM Encoder
        # Processes the observed trajectory
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # LSTM Decoder
        # Generates the predicted trajectory step-by-step
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layer
        # Maps the hidden state back to 2D coordinates
        self.fc = nn.Linear(hidden_dim, 2)
        
        self.relu = nn.ReLU()

    def forward(self, obs_traj, pred_len=12):
        """
        Forward pass for the model.
        
        Args:
        - obs_traj: Tensor of shape (batch_size, obs_len, 2)
        - pred_len: The length of the future trajectory to predict.
        
        Returns:
        - A tensor of shape (batch_size, pred_len, 2) representing the predicted trajectory.
        """
        # Get the batch size
        batch_size = obs_traj.size(0)

        # 1. Encode the observed trajectory
        # Embed the input coordinates
        embedded_obs = self.relu(self.embedding(obs_traj))
        
        # Pass through the encoder. We only need the final hidden and cell states.
        _, (hidden_state, cell_state) = self.encoder(embedded_obs)

        # 2. Decode to predict the future trajectory (autoregressive)
        # Initialize the list to store predictions
        predictions = []
        
        # Use the last observed position as the first input to the decoder
        last_obs_pos = obs_traj[:, -1, :]
        decoder_input = self.relu(self.embedding(last_obs_pos))
        
        # Unsqueeze to add a sequence length dimension of 1
        decoder_input = decoder_input.unsqueeze(1)

        # Loop for the length of the prediction
        for _ in range(pred_len):
            # Pass the input and hidden states through the decoder
            output, (hidden_state, cell_state) = self.decoder(decoder_input, (hidden_state, cell_state))
            
            # Get the predicted coordinate from the output
            pred_pos = self.fc(output.squeeze(1))
            predictions.append(pred_pos)
            
            # The next input to the decoder is the current prediction
            decoder_input = self.relu(self.embedding(pred_pos))
            decoder_input = decoder_input.unsqueeze(1)

        # Stack the predictions along the sequence dimension
        final_predictions = torch.stack(predictions, dim=1)
        
        return final_predictions