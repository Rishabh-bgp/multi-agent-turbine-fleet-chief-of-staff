import torch
import torch.nn as nn


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, n_features, conv=32, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_features, conv, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(conv, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1); self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        a = torch.softmax(self.att(h), 1)
        return self.head((a * h).sum(1)).squeeze(-1)


def power_curve(ws, rated_kw=2000.0, cut_in=3.0, rated=12.0, cut_out=25.0):
    import numpy as np
    ws = np.asarray(ws, float)
    p = np.where((ws >= cut_in) & (ws < rated), rated_kw * ((ws - cut_in) / (rated - cut_in)) ** 3, 0.0)
    return np.where((ws >= rated) & (ws < cut_out), rated_kw, p)
