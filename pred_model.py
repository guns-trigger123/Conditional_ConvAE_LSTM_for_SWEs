from torch import nn


class LSTMPredictor(nn.Module):
    def __init__(self):
        super(LSTMPredictor, self).__init__()
        input_size = 128
        hidden_size = 256
        self.predict_timestep = 5

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=5,
            dropout=0.4,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, input_size * self.predict_timestep)

    def forward(self, x):
        batch_size, fragment_length, latent_dim_AE = x.shape
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        out = out.reshape(batch_size, fragment_length, latent_dim_AE)
        return out


class ConditionalLSTMPredictor(nn.Module):
    def __init__(self):
        super(ConditionalLSTMPredictor, self).__init__()
        input_size = 128
        hidden_size = 256
        self.num_conditional = 2
        self.predict_timestep = 5

        self.lstm = nn.LSTM(
            input_size=input_size + self.num_conditional,
            hidden_size=hidden_size,
            num_layers=5,
            dropout=0.4,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, input_size * self.predict_timestep)

    def forward(self, x, condition_Hp):
        batch_size, fragment_length, latent_dim_AE = x.shape
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out * condition_Hp)
        out = out.reshape(batch_size, fragment_length, latent_dim_AE - self.num_conditional)
        return out
