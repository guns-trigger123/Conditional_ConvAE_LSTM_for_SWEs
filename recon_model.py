from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        input_channels, image_height, image_width = 3, 128, 128
        latent_dim = 128

        # Encoder
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
        )

        self.fc1 = nn.Linear(128 * (image_height // 16) * (image_width // 16), latent_dim)

        self.fc2 = nn.Linear(latent_dim, 128 * (image_height // 16) * (image_width // 16))

        # Encoder
        self.decode = nn.Sequential(
            nn.Unflatten(1, (128, image_height // 16, image_width // 16)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder(self, x):
        x = self.encode(x)
        x = self.fc1(x)
        return x

    def decoder(self, x):
        x = self.fc2(x)
        x = self.decode(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim=128, output_dim=2):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        for layer in self.fcn:
            x = layer(x)
        return x
