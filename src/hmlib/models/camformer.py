import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding function to inject time-related information into the model.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)  # Not a parameter, but saved with the model

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class RinkFeatureExtractor(nn.Module):
    """
    Extracts features from the ice rink segmentation image using a simple CNN.
    """

    def __init__(self, d_rink_feature):
        super(RinkFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Assuming RGB images
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )
        self.fc = nn.Linear(32, d_rink_feature)

    def forward(self, x):
        # x: [batch_size, 3, H, W]
        x = self.cnn(x)  # [batch_size, 32, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 32]
        x = self.fc(x)  # [batch_size, d_rink_feature]
        return x


class CameraPredictor(nn.Module):
    """
    Multi-transformer model that predicts camera focus point and bounding box size.
    Handles variable number of players using padding and masking.
    """

    def __init__(
        self,
        d_model,
        d_rink_feature,
        nhead_player,
        nhead_time,
        num_layers_player,
        num_layers_time,
        seq_len_past,
        seq_len_future,
    ):
        super(CameraPredictor, self).__init__()
        # Embedding layers
        self.embedding = nn.Linear(2, d_model)  # Player positions to d_model dimensions
        self.rink_embedding = nn.Linear(
            d_rink_feature, d_model
        )  # Rink features to d_model dimensions

        # Transformer encoder layers for player positions
        encoder_layer_player = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead_player)
        self.transformer_encoder_player = nn.TransformerEncoder(
            encoder_layer_player, num_layers=num_layers_player
        )

        # Positional encoding for time dimension
        self.pos_encoder_time = PositionalEncoding(d_model)

        # Transformer encoder for temporal sequence
        encoder_layer_time = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead_time)
        self.transformer_encoder_time = nn.TransformerEncoder(
            encoder_layer_time, num_layers=num_layers_time
        )

        # Output layer to predict camera parameters
        self.fc_out = nn.Linear(d_model, 4)  # Predicts (x_center, y_center, width, height)

        # Rink feature extractor
        self.rink_feature_extractor = RinkFeatureExtractor(d_rink_feature)

        self.seq_len_past = seq_len_past
        self.seq_len_future = seq_len_future

    def forward(self, x, x_lengths, rink_image):
        """
        x: List of player positions [batch_size, seq_len_past, variable_num_players, 2]
        x_lengths: [batch_size, seq_len_past] Number of players at each time step
        rink_image: [batch_size, 3, H, W]
        """
        batch_size, seq_len_past = x.size(0), x.size(1)

        # Extract and embed rink features
        rink_features = self.rink_feature_extractor(rink_image)  # [batch_size, d_rink_feature]
        rink_emb = (
            self.rink_embedding(rink_features).unsqueeze(1).unsqueeze(2)
        )  # [batch_size, 1, 1, d_model]

        # Prepare masks for variable number of players
        max_num_players = x.size(2)  # After padding
        player_masks = []
        x_emb_list = []

        for t in range(seq_len_past):
            x_t = x[:, t, :, :]  # [batch_size, num_players_t, 2]
            x_len_t = x_lengths[:, t]  # [batch_size]

            # Embedding
            x_t_emb = self.embedding(x_t)  # [batch_size, num_players_t, d_model]

            # Add rink embedding
            x_t_emb = x_t_emb + rink_emb[:, 0, 0, :].unsqueeze(
                1
            )  # [batch_size, num_players_t, d_model]

            # Create mask
            player_mask_t = (
                torch.arange(max_num_players).unsqueeze(0).expand(batch_size, -1).to(x.device)
            )
            player_mask_t = player_mask_t < x_len_t.unsqueeze(1)  # [batch_size, max_num_players]

            # Append for later processing
            x_emb_list.append(x_t_emb)
            player_masks.append(player_mask_t)

        # Process player positions at each time step
        x_players = []
        for t in range(seq_len_past):
            x_t_emb = x_emb_list[t]  # [batch_size, num_players_t, d_model]
            player_mask_t = player_masks[t]  # [batch_size, max_num_players]

            # Transpose for transformer input
            x_t_emb = x_t_emb.transpose(0, 1)  # [num_players_t, batch_size, d_model]

            # Create src_key_padding_mask
            src_key_padding_mask = ~player_mask_t  # Invert mask for padding positions

            # Handle variable sequence lengths by padding x_t_emb
            max_num_players = player_mask_t.size(1)
            if x_t_emb.size(0) < max_num_players:
                pad_size = max_num_players - x_t_emb.size(0)
                pad_emb = torch.zeros(pad_size, batch_size, x_t_emb.size(2)).to(x.device)
                x_t_emb = torch.cat(
                    [x_t_emb, pad_emb], dim=0
                )  # [max_num_players, batch_size, d_model]

            # Transformer encoder for players
            x_t_enc = self.transformer_encoder_player(
                x_t_emb, src_key_padding_mask=src_key_padding_mask
            )
            x_t_enc = x_t_enc.masked_fill(src_key_padding_mask.unsqueeze(2).transpose(0, 1), 0)

            # Aggregate over players
            x_t_mean = x_t_enc.sum(dim=0) / player_mask_t.sum(
                dim=1, keepdim=True
            )  # [batch_size, d_model]
            x_players.append(x_t_mean)

        # Stack and encode over time
        x_time = torch.stack(x_players, dim=0)  # [seq_len_past, batch_size, d_model]
        x_time = self.pos_encoder_time(x_time)
        x_time = self.transformer_encoder_time(x_time)  # [seq_len_past, batch_size, d_model]

        # Predict future camera parameters
        x_last = x_time[-1, :, :]  # [batch_size, d_model]
        x_future = x_last.unsqueeze(1).repeat(
            1, self.seq_len_future, 1
        )  # [batch_size, seq_len_future, d_model]
        out = self.fc_out(x_future)  # [batch_size, seq_len_future, 4]

        return out


def inference(model, positions, counts, rink_image, device):
    model.eval()
    with torch.no_grad():
        positions = positions.to(device)
        counts = counts.to(device)
        rink_image = rink_image.to(device)

        outputs = model(
            positions.unsqueeze(0), counts.unsqueeze(0), rink_image.unsqueeze(0)
        )  # Add batch dim
        outputs = outputs.squeeze(0)  # Remove batch dim
        return outputs  # [seq_len_future, 4]


# Example usage
def inference_test():
    # Assume positions, counts, rink_image are obtained from real data
    # For demonstration, we'll use a sample from the dataset

    sample_idx = 0
    positions_list, counts_sample, rink_image_sample, _ = dataset[sample_idx]

    # Pad positions and counts to max_num_players
    max_num_players_in_sample = counts_sample.max().item()
    positions_padded = []
    for t in range(seq_len_past):
        pos_t = positions_list[t]
        num_players_t = pos_t.size(0)
        if num_players_t < max_num_players_in_sample:
            pad_size = max_num_players_in_sample - num_players_t
            pad = torch.zeros(pad_size, 2)
            pos_t = torch.cat([pos_t, pad], dim=0)
        positions_padded.append(pos_t)
    positions_padded = torch.stack(
        positions_padded, dim=0
    )  # [seq_len_past, max_num_players_in_sample, 2]

    # Convert counts to tensor
    counts_sample = counts_sample

    # Perform inference
    outputs = inference(model, positions_padded, counts_sample, rink_image_sample, device)
    print("Predicted camera parameters:", outputs)

#
# TRAINING
#

# Include the previously defined classes here (PositionalEncoding, RinkFeatureExtractor, CameraPredictor)
# For brevity, I'll assume these classes are already defined as per the previous code.


# Simulated Dataset
class HockeyDataset(Dataset):

    def __init__(self, num_samples, seq_len_past, seq_len_future, max_num_players):
        super(HockeyDataset, self).__init__()
        self.num_samples = num_samples
        self.seq_len_past = seq_len_past
        self.seq_len_future = seq_len_future
        self.max_num_players = max_num_players  # Maximum possible players in the dataset

        # Simulate variable player counts
        self.player_counts = torch.randint(
            5, max_num_players + 1, (num_samples, seq_len_past)
        )  # At least 5 players

        # Simulate player positions
        self.player_positions = []
        for i in range(num_samples):
            positions = []
            for t in range(seq_len_past):
                num_players_t = self.player_counts[i, t]
                # Positions for existing players
                pos_t = torch.randn(num_players_t, 2) * 100  # [num_players_t, 2]
                positions.append(pos_t)
            self.player_positions.append(positions)  # List of lists

        # Simulate rink images (random noise)
        self.rink_images = torch.randn(num_samples, 3, 224, 224)

        # Simulate camera parameters (x_center, y_center, width, height)
        self.camera_params = torch.randn(num_samples, seq_len_future, 4) * 100

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        positions = self.player_positions[idx]  # List of [num_players_t, 2] for t in seq_len_past
        player_counts = self.player_counts[idx]  # [seq_len_past]
        rink_image = self.rink_images[idx]  # [3, 224, 224]
        camera_params = self.camera_params[idx]  # [seq_len_future, 4]
        return positions, player_counts, rink_image, camera_params


def collate_fn(batch):
    positions_batch, counts_batch, rink_images_batch, camera_params_batch = zip(*batch)
    batch_size = len(positions_batch)
    seq_len_past = len(positions_batch[0])

    # Find max number of players in the batch
    max_num_players = max(
        [counts_batch[i][t].item() for i in range(batch_size) for t in range(seq_len_past)]
    )

    # Pad positions and create position tensors
    positions_padded = []
    counts_padded = torch.stack(counts_batch)  # [batch_size, seq_len_past]
    for i in range(batch_size):
        positions_i = []
        for t in range(seq_len_past):
            pos_t = positions_batch[i][t]  # [num_players_t, 2]
            num_players_t = pos_t.size(0)
            # Pad to max_num_players
            if num_players_t < max_num_players:
                pad_size = max_num_players - num_players_t
                pad = torch.zeros(pad_size, 2)
                pos_t = torch.cat([pos_t, pad], dim=0)
            positions_i.append(pos_t)
        positions_i = torch.stack(positions_i, dim=0)  # [seq_len_past, max_num_players, 2]
        positions_padded.append(positions_i)

    positions_padded = torch.stack(
        positions_padded, dim=0
    )  # [batch_size, seq_len_past, max_num_players, 2]

    rink_images_batch = torch.stack(rink_images_batch)  # [batch_size, 3, H, W]
    camera_params_batch = torch.stack(camera_params_batch)  # [batch_size, seq_len_future, 4]

    return positions_padded, counts_padded, rink_images_batch, camera_params_batch


# Training function
# Training function remains the same, but with updated inputs
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for positions, counts, rink_images, camera_params in dataloader:
            positions = positions.to(device)
            counts = counts.to(device)
            rink_images = rink_images.to(device)
            camera_params = camera_params.to(device)

            optimizer.zero_grad()
            outputs = model(positions, counts, rink_images)  # [batch_size, seq_len_future, 4]

            loss = criterion(outputs, camera_params)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * positions.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Example usage
def train_test():
    # Define model parameters
    d_model = 128  # Embedding size
    d_rink_feature = 64  # Rink feature size
    nhead_player = 4
    nhead_time = 4
    num_layers_player = 2
    num_layers_time = 2
    seq_len_past = 5
    seq_len_future = 3

    # Instantiate the model
    model = CameraPredictor(
        d_model,
        d_rink_feature,
        nhead_player,
        nhead_time,
        num_layers_player,
        num_layers_time,
        seq_len_past,
        seq_len_future,
    )

    # Hyperparameters
    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-4

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset and dataloader
    max_num_players = 10  # Maximum possible players
    dataset = HockeyDataset(
        num_samples=1000,
        seq_len_past=seq_len_past,
        seq_len_future=seq_len_future,
        max_num_players=max_num_players,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs, device)
