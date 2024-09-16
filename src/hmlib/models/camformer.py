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
    """

    def __init__(
        self,
        num_players,
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

    def forward(self, x, rink_image):
        # x: [batch_size, seq_len_past, num_players, 2]
        # rink_image: [batch_size, 3, H, W]
        batch_size, seq_len_past, num_players, _ = x.size()

        # Flatten player positions and embed
        x = x.view(-1, 2)  # [batch_size * seq_len_past * num_players, 2]
        x = self.embedding(x)  # [batch_size * seq_len_past * num_players, d_model]
        x = x.view(
            batch_size, seq_len_past, num_players, -1
        )  # [batch_size, seq_len_past, num_players, d_model]

        # Extract and embed rink features
        rink_features = self.rink_feature_extractor(rink_image)  # [batch_size, d_rink_feature]
        rink_emb = (
            self.rink_embedding(rink_features).unsqueeze(1).unsqueeze(2)
        )  # [batch_size, 1, 1, d_model]
        rink_emb = rink_emb.expand(-1, seq_len_past, num_players, -1)  # Match dimensions

        # Combine player embeddings with rink embeddings
        x = x + rink_emb  # [batch_size, seq_len_past, num_players, d_model]

        # Process player positions at each time step
        x_players = []
        for t in range(seq_len_past):
            x_t = x[:, t, :, :].transpose(0, 1)  # [num_players, batch_size, d_model]
            x_t = self.transformer_encoder_player(x_t)  # [num_players, batch_size, d_model]
            x_t = x_t.mean(dim=0)  # Aggregate over players: [batch_size, d_model]
            x_players.append(x_t)

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


# Example usage
def inference_test():
    # Define model parameters
    num_players = 10  # Number of players on the ice
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
        num_players,
        d_model,
        d_rink_feature,
        nhead_player,
        nhead_time,
        num_layers_player,
        num_layers_time,
        seq_len_past,
        seq_len_future,
    )

    # Create dummy inputs
    batch_size = 2
    x = torch.randn(batch_size, seq_len_past, num_players, 2)  # Player positions
    rink_image = torch.randn(batch_size, 3, 224, 224)  # Ice rink segmentation images

    # Forward pass
    output = model(x, rink_image)  # [batch_size, seq_len_future, 4]
    print("Predicted camera parameters:", output)


#
# TRAINING
#

# Include the previously defined classes here (PositionalEncoding, RinkFeatureExtractor, CameraPredictor)
# For brevity, I'll assume these classes are already defined as per the previous code.


# Simulated Dataset
class HockeyDataset(Dataset):
    def __init__(self, num_samples, seq_len_past, seq_len_future, num_players):
        super(HockeyDataset, self).__init__()
        self.num_samples = num_samples
        self.seq_len_past = seq_len_past
        self.seq_len_future = seq_len_future
        self.num_players = num_players

        # Simulate player positions (x, y) within the rink dimensions (assuming 0 to 100 units)
        self.player_positions = torch.randn(num_samples, seq_len_past, num_players, 2) * 100

        # Simulate rink images (random noise for this example)
        self.rink_images = torch.randn(num_samples, 3, 224, 224)

        # Simulate camera parameters (x_center, y_center, width, height)
        self.camera_params = torch.randn(num_samples, seq_len_future, 4) * 100

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        positions = self.player_positions[idx]  # [seq_len_past, num_players, 2]
        rink_image = self.rink_images[idx]  # [3, 224, 224]
        camera_params = self.camera_params[idx]  # [seq_len_future, 4]
        return positions, rink_image, camera_params


# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for positions, rink_images, camera_params in dataloader:
            positions = positions.to(device)
            rink_images = rink_images.to(device)
            camera_params = camera_params.to(device)

            optimizer.zero_grad()
            outputs = model(positions, rink_images)  # [batch_size, seq_len_future, 4]

            loss = criterion(outputs, camera_params)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * positions.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Example usage
def train_test():
    # Define model parameters
    num_players = 10  # Number of players on the ice
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
        num_players,
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
    dataset = HockeyDataset(
        num_samples=1000,
        seq_len_past=seq_len_past,
        seq_len_future=seq_len_future,
        num_players=num_players,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs, device)
