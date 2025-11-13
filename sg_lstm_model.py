import torch
import torch.nn as nn

class SGLSTM(nn.Module):
    """
    A simplified Social Group LSTM (SG-LSTM) model for trajectory prediction.
    
    This "tiny reproduction" model fuses individual trajectory information with
    group trajectory information at the embedding level.
    """
    def __init__(self, embedding_dim=64, hidden_dim=64, num_layers=1):
        """
        Args:
        - embedding_dim: The dimension to embed individual and group coordinates.
        - hidden_dim: The dimension of the LSTM's hidden state.
        - num_layers: The number of layers in the LSTM.
        """
        super(SGLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Input embedding layer for individual trajectory
        self.ind_embedding = nn.Linear(2, embedding_dim)
        
        # Input embedding layer for group trajectory
        self.group_embedding = nn.Linear(2, embedding_dim)
        
        # The combined embedding will have size 2 * embedding_dim
        encoder_input_dim = 2 * embedding_dim
        
        # LSTM Encoder
        # Processes the *combined* trajectory
        self.encoder = nn.LSTM(encoder_input_dim, hidden_dim, num_layers, batch_first=True)
        
        # LSTM Decoder
        # Generates the predicted *individual* trajectory step-by-step
        # Note: The decoder input will just be the individual's embedded position
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layer
        # Maps the hidden state back to 2D coordinates
        self.fc = nn.Linear(hidden_dim, 2)
        
        self.relu = nn.ReLU()

    def forward(self, obs_traj, obs_group_traj, pred_len=12):
        """
        Forward pass for the model.
        
        Args:
        - obs_traj: Individual's observed trajectory (batch_size, obs_len, 2)
        - obs_group_traj: Group's observed trajectory (batch_size, obs_len, 2)
        - pred_len: The length of the future trajectory to predict.
        
        Returns:
        - A tensor of shape (batch_size, pred_len, 2) representing the predicted trajectory.
        """
        batch_size = obs_traj.size(0)

        # 1. Encode the observed trajectory
        # Embed both individual and group coordinates
        embedded_obs = self.relu(self.ind_embedding(obs_traj))
        embedded_group = self.relu(self.group_embedding(obs_group_traj))
        
        # Concatenate the embeddings
        combined_embedding = torch.cat((embedded_obs, embedded_group), dim=2)
        
        # Pass through the encoder. We only need the final hidden and cell states.
        # These states now contain information from BOTH individual and group paths.
        _, (hidden_state, cell_state) = self.encoder(combined_embedding)

        # 2. Decode to predict the future trajectory (autoregressive)
        predictions = []
        
        # Use the last observed *individual* position as the first input to the decoder
        last_obs_pos = obs_traj[:, -1, :]
        decoder_input = self.relu(self.ind_embedding(last_obs_pos))
        decoder_input = decoder_input.unsqueeze(1) # Add sequence dim

        for _ in range(pred_len):
            # Pass the input and hidden states through the decoder
            output, (hidden_state, cell_state) = self.decoder(decoder_input, (hidden_state, cell_state))
            
            # Get the predicted coordinate from the output
            pred_pos = self.fc(output.squeeze(1))
            predictions.append(pred_pos)
            
            # The next input to the decoder is the current prediction (embedded)
            decoder_input = self.relu(self.ind_embedding(pred_pos))
            decoder_input = decoder_input.unsqueeze(1)

        # Stack the predictions
        final_predictions = torch.stack(predictions, dim=1)
        
        return final_predictions